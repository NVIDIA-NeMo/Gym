# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny catch-all DNS server for network-isolated prebuilt Apex worlds.

Prebuilt worlds sometimes refer to in-sandbox web apps by hostnames such
as erpnext.io or wikijs.local. In an Apptainer private network namespace those
names have no upstream DNS, even though the services live on loopback. This
resolver answers all A and AAAA questions with loopback addresses.
"""

from __future__ import annotations

import argparse
import ipaddress
import socketserver
import struct
from typing import Iterable


QTYPE_A = 1
QTYPE_AAAA = 28
QTYPE_ANY = 255
QCLASS_IN = 1
QCLASS_ANY = 255


def _read_name(packet: bytes, offset: int) -> int:
    """Return the offset after a DNS name, following compression pointers."""
    jumped = False
    seen: set[int] = set()
    while True:
        if offset >= len(packet):
            raise ValueError("truncated DNS name")
        length = packet[offset]
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(packet):
                raise ValueError("truncated DNS pointer")
            pointer = ((length & 0x3F) << 8) | packet[offset + 1]
            if pointer in seen:
                raise ValueError("recursive DNS pointer")
            seen.add(pointer)
            if not jumped:
                offset += 2
                jumped = True
            offset = pointer
            continue
        offset += 1
        if length == 0:
            return offset
        offset += length


def _questions(packet: bytes, qdcount: int) -> tuple[bytes, list[tuple[int, int]]]:
    offset = 12
    question_end = offset
    parsed: list[tuple[int, int]] = []
    for _ in range(qdcount):
        question_end = _read_name(packet, question_end)
        if question_end + 4 > len(packet):
            raise ValueError("truncated DNS question")
        qtype, qclass = struct.unpack("!HH", packet[question_end : question_end + 4])
        parsed.append((qtype, qclass))
        question_end += 4
    return packet[12:question_end], parsed


def _answer(qtype: int, address: str) -> bytes:
    payload = ipaddress.ip_address(address).packed
    return b"\xc0\x0c" + struct.pack("!HHIH", qtype, QCLASS_IN, 0, len(payload)) + payload


def _answers(questions: Iterable[tuple[int, int]]) -> list[bytes]:
    answers: list[bytes] = []
    for qtype, qclass in questions:
        if qclass not in {QCLASS_IN, QCLASS_ANY}:
            continue
        if qtype in {QTYPE_A, QTYPE_ANY}:
            answers.append(_answer(QTYPE_A, "127.0.0.1"))
        if qtype in {QTYPE_AAAA, QTYPE_ANY}:
            answers.append(_answer(QTYPE_AAAA, "::1"))
    return answers


def build_response(packet: bytes) -> bytes:
    if len(packet) < 12:
        raise ValueError("truncated DNS header")
    request_id, flags, qdcount, _ancount, _nscount, _arcount = struct.unpack("!HHHHHH", packet[:12])
    question_blob, parsed_questions = _questions(packet, qdcount)
    answers = _answers(parsed_questions)
    # QR=1, AA=1, copy RD, RA=0, RCODE=0. This is authoritative for the sandbox.
    response_flags = 0x8400 | (flags & 0x0100)
    header = struct.pack("!HHHHHH", request_id, response_flags, qdcount, len(answers), 0, 0)
    return header + question_blob + b"".join(answers)


class DNSHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        packet, sock = self.request
        try:
            response = build_response(packet)
        except Exception:
            if len(packet) >= 2:
                response = packet[:2] + b"\x84\x02\x00\x00\x00\x00\x00\x00\x00\x00"
            else:
                return
        sock.sendto(response, self.client_address)


class UDPServer(socketserver.ThreadingUDPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=53)
    args = parser.parse_args()
    with UDPServer((args.host, args.port), DNSHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
