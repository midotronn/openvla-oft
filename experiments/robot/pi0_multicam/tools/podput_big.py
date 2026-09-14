#!/usr/bin/env python3
"""Reliable large-file push over the runpod PTY proxy. Sends base64 in small chunks,
looping send() (the original podput used a single send() -> truncated >window). md5-verified.
Usage: podput_big.py LOCAL REMOTE"""
import os, sys, time, base64, hashlib, paramiko
USER=os.environ.get("POD_USER","j5dgev6ahqr8t3-64412155"); HOST="ssh.runpod.io"
KEY=os.path.expanduser("~/.ssh/id_ed25519")
local,remote=sys.argv[1],sys.argv[2]
data=open(local,"rb").read()
md5=hashlib.md5(data).hexdigest()
b64=base64.b64encode(data).decode()   # single line, chars are [A-Za-z0-9+/=]
key=paramiko.Ed25519Key.from_private_key_file(KEY)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST,username=USER,pkey=key,timeout=60,banner_timeout=60,auth_timeout=60)
sh=c.invoke_shell(width=220,height=50);time.sleep(2)
if sh.recv_ready(): sh.recv(1<<20)

def send_all(s):
    d=s.encode()
    while d:
        n=sh.send(d)
        d=d[n:]
        if sh.recv_ready(): sh.recv(1<<20)   # drain echo to avoid remote buffer stall

# disable echo + bracketed paste, reset temp file
send_all("stty -echo 2>/dev/null; printf '\\033[?2004l'; rm -f /tmp/_pb64; echo RESET_OK\n")
time.sleep(1)
if sh.recv_ready(): sh.recv(1<<20)

CH=800
for i in range(0,len(b64),CH):
    send_all(f"printf %s '{b64[i:i+CH]}' >> /tmp/_pb64\n")
    time.sleep(0.02)

send_all(f"base64 -d /tmp/_pb64 > {remote} && echo PUT_OK_$(wc -c < {remote})_$(md5sum {remote} | cut -d' ' -f1)\n")
buf="";t=time.time()
while time.time()-t<120:
    if sh.recv_ready(): buf+=sh.recv(1<<20).decode("utf-8","replace")
    if "PUT_OK_" in buf: break
    time.sleep(0.3)
c.close()
line=[l for l in buf.splitlines() if "PUT_OK_" in l and "echo" not in l]
print("local_bytes",len(data),"local_md5",md5)
print("remote:",line[-1].strip() if line else "(no PUT_OK)")
ok=any(md5 in l for l in line)
print("VERIFIED" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
