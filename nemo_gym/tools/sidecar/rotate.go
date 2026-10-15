// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"io"
	"log"
	"math/rand/v2"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// An upstream load balancer closes every client connection once it reaches a
// fixed age (AWS ALB: 3600s by default) by sending a graceful GOAWAY. Requests
// that were already written to the connection can only be re-sent when their
// body is buffered (see -retry-body-limit); larger ones fail. rotatingTransport
// avoids the event instead: before the limit, new requests move to a fresh
// transport (and so a fresh connection), while requests already in flight
// finish on the old one, which is then closed.
//
// A connection never outlives the transport that opened it, so rotating whole
// transports bounds every connection's age without tracking connections.

type generation struct {
	transport *http.Transport
	inflight  atomic.Int64
	retired   atomic.Bool
}

type rotatingTransport struct {
	newTransport func() *http.Transport

	mu      sync.RWMutex
	current *generation
}

func newRotatingTransport(newTransport func() *http.Transport) *rotatingTransport {
	return &rotatingTransport{newTransport: newTransport, current: &generation{transport: newTransport()}}
}

func (r *rotatingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Count the request while holding the read lock, so rotate() either sees it
	// in flight or hands it the new generation.
	r.mu.RLock()
	g := r.current
	g.inflight.Add(1)
	r.mu.RUnlock()

	resp, err := g.transport.RoundTrip(req)
	if err != nil {
		r.release(g)
		return nil, err
	}
	// The request is in flight until its response body is closed, which is also
	// how streaming (SSE) responses keep their generation alive.
	resp.Body = &releasingBody{ReadCloser: resp.Body, release: func() { r.release(g) }}
	return resp, nil
}

func (r *rotatingTransport) release(g *generation) {
	if g.inflight.Add(-1) == 0 && g.retired.Load() {
		g.transport.CloseIdleConnections()
	}
}

// rotate installs a fresh transport and retires the previous one.
func (r *rotatingTransport) rotate() {
	fresh := &generation{transport: r.newTransport()}
	r.mu.Lock()
	old := r.current
	r.current = fresh
	// Set before inflight is read below, so either this call or the last
	// release() closes the old transport's connections.
	old.retired.Store(true)
	r.mu.Unlock()
	if old.inflight.Load() == 0 {
		old.transport.CloseIdleConnections()
	}
}

// rotateEvery rotates once per maxAge, each time after 90-100% of it, so that
// several sidecars started together do not retire their connections together.
func (r *rotatingTransport) rotateEvery(maxAge time.Duration, stop <-chan struct{}) {
	for {
		wait := time.Duration(float64(maxAge) * (0.9 + 0.1*rand.Float64()))
		timer := time.NewTimer(wait)
		select {
		case <-stop:
			timer.Stop()
			return
		case <-timer.C:
		}
		r.rotate()
		log.Printf("retired upstream connections after %v; new requests use a fresh connection", wait.Round(time.Second))
	}
}

type releasingBody struct {
	io.ReadCloser
	release func()
	once    sync.Once
}

func (b *releasingBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.release)
	return err
}
