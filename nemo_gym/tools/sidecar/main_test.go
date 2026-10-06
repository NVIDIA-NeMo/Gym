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
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// Builds the sidecar and runs it against a local fake HTTP/2 origin. It never
// sends traffic or credentials anywhere else.
func TestSidecarForwardsOverHTTP2AndDrains(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "h2-ping-sidecar-test")
	if out, err := exec.Command("go", "build", "-buildvcs=false", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}

	started, release := make(chan struct{}), make(chan struct{})
	origin := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor != 2 || r.Header.Get("Authorization") != "Bearer local-test-only" {
			http.Error(w, "protocol/header mismatch", 500)
			return
		}
		if r.URL.Path == "/delayed" {
			close(started)
			<-release
		}
		w.Header().Set("X-Upstream-Protocol", r.Proto)
		_, _ = io.Copy(w, r.Body)
	}))
	origin.EnableHTTP2 = true
	origin.StartTLS()
	defer origin.Close()
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := listener.Addr().String()
	_ = listener.Close()
	ready := filepath.Join(t.TempDir(), "ready")
	var logs bytes.Buffer
	cmd := exec.Command(bin, "-listen", addr, "-upstream", origin.URL,
		"-insecure-skip-verify", "-ping-interval", "50ms", "-shutdown-grace", "2s", "-ready-file", ready)
	cmd.Stdout, cmd.Stderr = &logs, &logs
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	defer func() { _ = cmd.Process.Kill() }()

	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("startup failed: %v\n%s", err, logs.String())
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("readiness timeout")
		}
		time.Sleep(10 * time.Millisecond)
	}
	pid, err := os.ReadFile(ready)
	if err != nil || strings.TrimSpace(string(pid)) != fmt.Sprint(cmd.Process.Pid) {
		t.Fatalf("incorrect readiness marker: %q %v", pid, err)
	}

	client := &http.Client{Timeout: 5 * time.Second}
	send := func(path, body string) error {
		req, err := http.NewRequest("POST", "http://"+addr+path, strings.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer local-test-only")
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		got, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		if resp.StatusCode != 200 || string(got) != body || resp.Header.Get("X-Upstream-Protocol") != "HTTP/2.0" {
			return fmt.Errorf("status=%d body=%q protocol=%s", resp.StatusCode, got, resp.Header.Get("X-Upstream-Protocol"))
		}
		return nil
	}
	for _, body := range []string{"short", strings.Repeat("x", 4096)} {
		if err := send("/echo", body); err != nil {
			t.Fatal(err)
		}
	}

	// An in-flight request must survive SIGTERM until the origin finishes,
	// rather than being cut off by process exit.
	result := make(chan error, 1)
	go func() { result <- send("/delayed", "drained") }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("origin did not start")
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)
	time.Sleep(100 * time.Millisecond)
	close(release)
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("shutdown timeout")
	}
	if _, err := os.Stat(ready); !os.IsNotExist(err) {
		t.Fatalf("ready file not removed on exit: %v", err)
	}
}

// startSidecar builds the sidecar, runs it against upstream and waits until it is ready.
func startSidecar(t *testing.T, upstream string, extraArgs ...string) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "h2-ping-sidecar-test")
	if out, err := exec.Command("go", "build", "-buildvcs=false", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := listener.Addr().String()
	_ = listener.Close()
	ready := filepath.Join(t.TempDir(), "ready")
	args := append([]string{"-listen", addr, "-upstream", upstream, "-insecure-skip-verify", "-ready-file", ready}, extraArgs...)
	var logs bytes.Buffer
	cmd := exec.Command(bin, args...)
	cmd.Stdout, cmd.Stderr = &logs, &logs
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _, _ = cmd.Process.Wait() })
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			return addr
		}
		if time.Now().After(deadline) {
			t.Fatalf("readiness timeout\n%s", logs.String())
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// When the upstream closes a connection with a graceful GOAWAY, a request that was
// already written to it must be re-sent on a fresh connection instead of failing with
// a 502. That needs the request body to be buffered so Request.GetBody exists.
func TestSidecarRetriesRequestsRefusedAfterGoaway(t *testing.T) {
	origin := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(w, r.Body)
	}))
	origin.EnableHTTP2 = true
	origin.StartTLS()
	defer origin.Close()
	// After each response the origin sends GOAWAY, as a load balancer does when it
	// retires a long-lived connection.
	origin.Config.SetKeepAlivesEnabled(false)

	run := func(extraArgs ...string) (failures int) {
		addr := startSidecar(t, origin.URL, extraArgs...)
		client := &http.Client{Timeout: 10 * time.Second}
		var wg sync.WaitGroup
		var mu sync.Mutex
		for i := 0; i < 40; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				body := strings.Repeat("x", 1024)
				resp, err := client.Post("http://"+addr+"/echo", "text/plain", strings.NewReader(body))
				ok := err == nil
				if ok {
					got, _ := io.ReadAll(resp.Body)
					resp.Body.Close()
					ok = resp.StatusCode == 200 && string(got) == body
				}
				if !ok {
					mu.Lock()
					failures++
					mu.Unlock()
				}
			}()
			time.Sleep(2 * time.Millisecond)
		}
		wg.Wait()
		return failures
	}

	t.Logf("failures without buffering (-retry-body-limit 0): %d", run("-retry-body-limit", "0"))
	if failures := run(); failures != 0 {
		t.Fatalf("%d of 40 requests failed even with body buffering", failures)
	}
}
