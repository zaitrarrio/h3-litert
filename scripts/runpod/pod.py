"""RunPod pods through GraphQL (api.runpod.io; rest.runpod.io is unreachable from
the sandbox), with the on-pod agent reachable through RunPod's HTTP proxy.

    pod.py create NAME GPU_TYPE DC VOLUME DISK_GB HOURS   -> POD <id> <$/h> <gpu> <token>
    pod.py status POD_ID
    pod.py terminate POD_ID
    pod.py list

`create` boots nvidia/cuda:12.8.1-runtime-ubuntu24.04, installs python3 and
curl, fetches scripts/runpod/agent.py from this repository's branch on GitHub
and serves it on port 8000 (`ports: "8000/http"`), with a backstop that
terminates the pod after HOURS through the pod-scoped RUNPOD_API_KEY RunPod
injects (an account key is never put in a pod). The volume mounts at
/workspace. Needs $RUNPOD_API_KEY.
"""
import json
import os
import secrets
import shlex
import sys
import urllib.request

KEY = os.environ["RUNPOD_API_KEY"]
BRANCH = os.environ.get("H3_LITERT_BRANCH", "claude/wonderful-ride-ag64z7")
RAW = f"https://raw.githubusercontent.com/zaitrarrio/h3-litert/{BRANCH}/scripts/runpod/agent.py"


def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    if r.get("errors"):
        sys.exit(f"GraphQL: {json.dumps(r['errors'])[:800]}")
    return r["data"]


def boot(token, hours):
    lines = [
        "set -u", "exec >>/root/boot.log 2>&1", 'echo "== boot $(date -u) pod $RUNPOD_POD_ID"',
        "nvidia-smi; free -g; nproc; df -h / /workspace",
        'terminate() { curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" '
        '-d "{\\"query\\":\\"mutation { podTerminate(input: {podId: \\\\\\"$RUNPOD_POD_ID\\\\\\"}) }\\"}" https://api.runpod.io/graphql; }',
        "export -f terminate",
        f'(sleep {int(hours * 3600)}; echo "backstop: terminating after {hours} h"; while :; do terminate; sleep 60; done) &',
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update -qq && apt-get install -y -qq curl ca-certificates python3 python3-venv python3-pip git ffmpeg "
        "ocl-icd-libopencl1 clinfo build-essential pkg-config libssl-dev >/dev/null && echo 'apt ok'",
        "mkdir -p /etc/OpenCL/vendors && echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd",
        f"curl -sfL -o /root/agent.py {shlex.quote(RAW)} && echo 'agent ok'",
        f"python3 /root/agent.py {token} 8000 &",
        "echo READY", "sleep infinity",
    ]
    return "\n".join(lines)


op = sys.argv[1]
s = json.dumps
if op == "create":
    name, gpu, dc, vol, disk, hours = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), float(sys.argv[7])
    token = secrets.token_hex(12)
    start = "bash -c " + shlex.quote(boot(token, hours))
    volume = f'networkVolumeId: {s(vol)}, volumeMountPath: "/workspace",' if vol != "-" else ""
    dcf = f"dataCenterId: {s(dc)}," if dc != "-" else ""
    q = f"""mutation {{ podFindAndDeployOnDemand(input: {{
      cloudType: SECURE, gpuCount: 1, gpuTypeId: {s(gpu)}, {dcf}
      name: {s(name)}, imageName: "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
      dockerArgs: {s(start)}, containerDiskInGb: {disk}, volumeInGb: 0, ports: "8000/http",
      {volume}
      env: [{{key: "NVIDIA_DRIVER_CAPABILITIES", value: "all"}}]
    }}) {{ id costPerHr machine {{ gpuDisplayName }} }} }}"""
    p = gql(q)["podFindAndDeployOnDemand"]
    print("POD", p["id"], p.get("costPerHr"), p["machine"]["gpuDisplayName"], token)
elif op == "status":
    print(json.dumps(gql(f'{{ pod(input: {{podId: {s(sys.argv[2])}}}) {{ id name desiredStatus costPerHr runtime {{ uptimeInSeconds ports {{ ip isIpPublic privatePort publicPort type }} }} machine {{ gpuDisplayName }} }} }}')["pod"]))
elif op == "terminate":
    print(gql(f'mutation {{ podTerminate(input: {{podId: {s(sys.argv[2])}}}) }}'))
elif op == "list":
    for p in gql("{ myself { pods { id name desiredStatus costPerHr } } }")["myself"]["pods"]:
        print(p["id"], p["desiredStatus"], p["costPerHr"], p["name"])
else:
    sys.exit(__doc__)
