#!/usr/bin/env python3
"""Correct long-keepalive for the runpod PTY proxy. Disables terminal echo so the command line
(which contains the completion sentinel) is NOT echoed back and mistaken for completion. Streams
output; detects completion via sentinel+exit-digit at line start. Holds ONE channel open (keeps
pod awake). Usage: podhold.py "CMD"   (env POD_TIMEOUT seconds, default 3000)."""
import os, sys, time, uuid, re, paramiko
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
USER=os.environ.get("POD_USER","j5dgev6ahqr8t3-64412155"); HOST=os.environ.get("POD_HOST","ssh.runpod.io")
KEY=os.path.expanduser(os.environ.get("POD_KEY","~/.ssh/id_ed25519"))
TIMEOUT=int(os.environ.get("POD_TIMEOUT","3000"))
cmd=" ".join(sys.argv[1:]) if len(sys.argv)>1 else sys.stdin.read()
end="END_"+uuid.uuid4().hex
key=paramiko.Ed25519Key.from_private_key_file(KEY)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST,username=USER,pkey=key,timeout=60,banner_timeout=60,auth_timeout=60)
sh=c.invoke_shell(width=200,height=60);time.sleep(2)
if sh.recv_ready(): sh.recv(1<<20)
# disable echo + bracketed paste so the command line isn't echoed back
sh.send("stty -echo; printf '\\033[?2004l'\n");time.sleep(1)
if sh.recv_ready(): sh.recv(1<<20)
sh.send(f"{cmd}; echo {end}_$?\n")
buf="";start=time.time()
pat=re.compile(re.escape(end)+r"_\d")
while True:
    if sh.recv_ready():
        chunk=sh.recv(1<<20).decode("utf-8","replace")
        chunk=re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]","",chunk); chunk=re.sub(r"\x1b\][0-9;]*\x07","",chunk)
        sys.stdout.write(chunk); sys.stdout.flush(); buf+=chunk
        if pat.search(buf): break
    else:
        time.sleep(0.5)
    if time.time()-start>TIMEOUT:
        sys.stdout.write("\n[podhold timeout]\n");break
try:
    sh.send("exit\n");time.sleep(0.3);c.close()
except Exception: pass
