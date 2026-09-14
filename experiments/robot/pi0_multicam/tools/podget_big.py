#!/usr/bin/env python3
"""Reliable large-file PULL over the runpod PTY proxy. Reads base64 in fixed byte-offset chunks
(tail -c +off | head -c N | base64 -w0), reassembles, md5-verified. Usage: podget_big.py REMOTE LOCAL"""
import os, sys, time, re, base64, hashlib, paramiko
USER=os.environ.get("POD_USER","j5dgev6ahqr8t3-64412155"); HOST="ssh.runpod.io"
KEY=os.path.expanduser("~/.ssh/id_ed25519")
remote,local=sys.argv[1],sys.argv[2]
key=paramiko.Ed25519Key.from_private_key_file(KEY)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST,username=USER,pkey=key,timeout=60,banner_timeout=60,auth_timeout=60)
sh=c.invoke_shell(width=220,height=50);time.sleep(2)
if sh.recv_ready(): sh.recv(1<<20)

def run(cmd, terminator):
    sh.send(cmd+"\n")
    buf="";t=time.time()
    while time.time()-t<120:
        if sh.recv_ready(): buf+=sh.recv(1<<20).decode("utf-8","replace")
        if terminator in buf: break
        time.sleep(0.1)
    return buf

sh.send("stty -echo 2>/dev/null; printf '\\033[?2004l'\n");time.sleep(0.6)
if sh.recv_ready(): sh.recv(1<<20)

# get size + md5
info=run(f"echo SZ_$(wc -c < {remote})_$(md5sum {remote} | cut -d' ' -f1)_END", "_END")
m=re.search(r"SZ_(\d+)_([0-9a-f]{32})_END", info)
size=int(m.group(1)); md5=m.group(2)
CH=24000  # raw bytes per chunk
data=bytearray()
off=0
while off<size:
    n=min(CH,size-off)
    beg="B%d"%off; end="E%d"%off
    out=run(f"printf {beg}; tail -c +{off+1} {remote} | head -c {n} | base64 -w0; printf {end}", end)
    seg=out.split(beg,1)[-1].split(end,1)[0]
    b64="".join(re.findall(r"[A-Za-z0-9+/=]", seg))
    b64=b64[:len(b64)-(len(b64)%4)]
    chunk=base64.b64decode(b64)
    data+=chunk
    off+=len(chunk)
    if len(chunk)==0:
        print("STALL at off",off);break
c.close()
open(local,"wb").write(data)
got=hashlib.md5(data).hexdigest()
print(f"remote_size {size} remote_md5 {md5}")
print(f"local_size {len(data)} local_md5 {got}")
print("VERIFIED" if got==md5 and len(data)==size else "MISMATCH")
sys.exit(0 if (got==md5 and len(data)==size) else 1)
