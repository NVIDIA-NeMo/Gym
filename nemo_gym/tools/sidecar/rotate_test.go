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
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// openConnListener counts the origin's accepted connections that are still open.
type openConnListener struct {
	net.Listener
	open atomic.Int64
}

func (l *openConnListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	l.open.Add(1)
	return &openConn{Conn: c, l: l}, nil
}

type openConn struct {
	net.Conn
	l    *openConnListener
	once sync.Once
}

func (c *openConn) Close() error {
	c.once.Do(func() { c.l.open.Add(-1) })
	return c.Conn.Close()
}

// A retired connection must be closed once its last request finishes, so rotation bounds
// connection age instead of leaving old connections open (and PINGed) indefinitely.
func TestSidecarRotationClosesRetiredConnections(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	origin := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/slow" {
			close(started)
			<-release
		}
		_, _ = io.WriteString(w, r.RemoteAddr)
	}))
	conns := &openConnListener{Listener: origin.Listener}
	origin.Listener = conns
	origin.EnableHTTP2 = true
	origin.StartTLS()
	t.Cleanup(origin.Close)
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	client := &http.Client{Timeout: 20 * time.Second}
	addr := startSidecar(t, origin.URL, "-max-conn-age", "300ms")

	slow := make(chan error, 1)
	go func() {
		resp, err := client.Get("http://" + addr + "/slow")
		if err == nil {
			_, err = io.ReadAll(resp.Body)
			resp.Body.Close()
		}
		slow <- err
	}()
	<-started
	time.Sleep(700 * time.Millisecond) // the slow request's connection is retired while in flight
	_ = get(t, client, "http://"+addr+"/addr")
	close(release)
	if err := <-slow; err != nil {
		t.Fatal(err)
	}

	// Nothing is in flight now, so every generation is retired within one more rotation and closes its connection.
	deadline := time.Now().Add(3 * time.Second)
	for conns.open.Load() != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("%d upstream connection(s) still open after their generations were retired", conns.open.Load())
		}
		time.Sleep(20 * time.Millisecond)
	}
}
