abi <abi/4.0>,
include <tunables/global>

profile bwrap-system /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap-system>
}

profile bwrap-codex /home/jetson/.codex/**/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap-codex>
}

profile codex-linux-sandbox /home/jetson/.codex/**/codex-linux-sandbox flags=(unconfined) {
  userns,
  include if exists <local/codex-linux-sandbox>
}
