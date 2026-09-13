#!/usr/bin/env bash
# Install the Intel userspace runtimes OpenVINO needs to see the iGPU and the
# NPU on a Core Ultra machine.  The kernel drivers (i915/xe, intel_vpu) already
# ship with Ubuntu; what is missing lives in userspace.
#
#   sudo bash scripts/setup_intel_runtime.sh            # install
#   sudo bash scripts/setup_intel_runtime.sh --dry-run  # download + report only
#
# Two things this is careful about, both learned the hard way:
#
#  * Intel's *latest* compute-runtime is built against glibc 2.38 / libstdc++ 13.
#    Ubuntu 22.04 has 2.35 / 12, so the newest release can never install here.
#    Versions are therefore pinned to the last ones that target 22.04, and every
#    downloaded package's libc6 requirement is checked before anything is run.
#  * `dpkg -i` followed by `apt-get -f install` will happily *remove* working
#    packages to resolve a conflict (it took out the VA-API video drivers).
#    Installing via apt with --no-remove aborts instead.
set -uo pipefail

DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
if [[ $EUID -ne 0 && $DRY -eq 0 ]]; then echo "run me with sudo"; exit 1; fi

CODENAME=$(. /etc/os-release && echo "${VERSION_ID}")
GLIBC=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')
echo "Ubuntu ${CODENAME}, glibc ${GLIBC}"

case "$CODENAME" in
  22.04)
    IGC=1.0.17537.20
    NEO=24.35.30872.22
    NEO_L0=1.3.30872.22
    GMM=22.5.0
    LZ=1.17.6 ; LZ_SUFFIX="+u22.04"
    NPU_TAG=v1.10.0 ; NPU_DISTRO=ubuntu22.04
    ;;
  24.04)
    IGC=1.0.17537.20
    NEO=24.35.30872.22
    NEO_L0=1.3.30872.22
    GMM=22.5.0
    LZ=1.17.6 ; LZ_SUFFIX="+u24.04"
    NPU_TAG=v1.10.0 ; NPU_DISTRO=ubuntu24.04
    ;;
  *) echo "unrecognised release '${CODENAME}' -- edit the pins in this script"; exit 1 ;;
esac

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT; cd "$TMP"
get() { echo "    $(basename "$1")"; curl -fsSLO "$1" || echo "    !! failed: $1"; }

echo "==> [1/4] graphics compiler + GPU compute runtime (pinned for ${CODENAME})"
IGCB=https://github.com/intel/intel-graphics-compiler/releases/download/igc-${IGC}
get "${IGCB}/intel-igc-core_${IGC}_amd64.deb"
get "${IGCB}/intel-igc-opencl_${IGC}_amd64.deb"
NEOB=https://github.com/intel/compute-runtime/releases/download/${NEO}
get "${NEOB}/intel-opencl-icd_${NEO}_amd64.deb"
get "${NEOB}/intel-level-zero-gpu_${NEO_L0}_amd64.deb"
get "${NEOB}/libigdgmm12_${GMM}_amd64.deb"

echo "==> [2/4] Level Zero loader"
get "https://github.com/oneapi-src/level-zero/releases/download/v${LZ}/level-zero_${LZ}${LZ_SUFFIX}_amd64.deb"

echo "==> [3/4] NPU driver (${NPU_TAG})"
# Asset names embed a build id, so resolve them from the release rather than guess.
JSON=$(curl -fsSL "https://api.github.com/repos/intel/linux-npu-driver/releases/tags/${NPU_TAG}" || true)
for pkg in intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu; do
  url=$(echo "$JSON" | grep -o '"browser_download_url": *"[^"]*"' | sed 's/.*: *"//;s/"$//' \
        | grep "/${pkg}_" | grep "${NPU_DISTRO}" | grep -v dbgsym | grep '\.deb$' | head -1)
  [[ -n "$url" ]] && get "$url" || echo "    !! no ${pkg} asset for ${NPU_DISTRO} in ${NPU_TAG}"
done

echo "==> [4/4] checking each package against this system before installing"
shopt -s nullglob
ok=(); skipped=()
for d in ./*.deb; do
  need=$(dpkg-deb -f "$d" Depends 2>/dev/null | grep -oE 'libc6 \(>= [0-9.]+\)' | grep -oE '[0-9.]+$' | sort -V | tail -1)
  if [[ -n "$need" ]] && ! dpkg --compare-versions "$GLIBC" ge "$need"; then
    echo "    SKIP $(basename "$d") -- needs glibc >= ${need}, system has ${GLIBC}"
    skipped+=("$(basename "$d")"); continue
  fi
  ok+=("$d")
done
echo "    ${#ok[@]} installable, ${#skipped[@]} skipped"

if (( DRY )); then echo "dry run -- nothing installed"; exit 0; fi
if (( ${#ok[@]} == 0 )); then echo "nothing installable; aborting"; exit 1; fi

apt-get update -qq 2>/dev/null || true
apt-get install -y -qq ocl-icd-libopencl1 clinfo || true
# --no-remove: abort rather than uninstall working packages to satisfy a dep
if ! apt-get install -y --no-remove "${ok[@]}"; then
  echo "!! apt refused the package set; nothing was changed. Re-run with --dry-run"
  echo "   to inspect, or file the output. System left as it was."
  exit 1
fi

for g in render video; do
  getent group "$g" >/dev/null && usermod -a -G "$g" "${SUDO_USER:-$USER}" || true
done
cat >/etc/udev/rules.d/10-intel-npu.rules <<'RULE'
SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"
RULE
udevadm control --reload-rules && udevadm trigger --subsystem-match=accel || true

echo
echo "Done.  LOG OUT AND BACK IN (group membership applies at login), then:"
echo "    clinfo -l"
echo "    .venv/bin/python -c \"import openvino as ov; print(ov.Core().available_devices)\""
echo "Expect ['CPU','GPU','NPU'].  Then:  make bench"
