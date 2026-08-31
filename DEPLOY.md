# Deploying

The short version, for when you have a version you want live.

```bash
git push                                    # CI builds and publishes the image
# wait for the "images" workflow to go green
DOMAIN=artificialintelligentduck.duckdns.org \
HOST=95.216.190.197 \
KEY=~/.ssh/hetzner \
./deploy.sh
```

That is the whole thing. The rest of this file is what those three lines do,
what to check, and what to do when one of them does not.

---

## The one rule

**The server runs a published image, not your working copy.**

`deploy.sh` copies the source up, but the containers start from
`ghcr.io/svaaan/helloworldai-server`, which is built by GitHub Actions. So:

> Code changes only reach the server after a push to `main` and a green
> `images` run.

Deploying without pushing first re-runs the *old* image against your new
compose files, and nothing about that looks wrong until you wonder why your fix
is not there. Check the commit on the box if you are unsure:

```bash
ssh -i ~/.ssh/hetzner ubuntu@95.216.190.197 \
  "docker inspect helloworldai-coordinator-1 --format '{{.Config.Image}}'"
```

The exception is `BUILD_ON_SERVER=1`, below.

---

## What you are deploying to

| | |
|---|---|
| Host | `95.216.190.197` — Hetzner CX23, Helsinki |
| Domain | `artificialintelligentduck.duckdns.org` |
| SSH | `ssh -i ~/.ssh/hetzner ubuntu@95.216.190.197` |
| Path | `~/HelloWorldAi` |
| Compose project | `helloworldai` |
| Image | `ghcr.io/svaaan/helloworldai-server:latest` |

Four containers: `caddy`, `helloworldai-coordinator-1`,
`helloworldai-dashboard-1`, `mongo_prod`.

Only Caddy is reachable from outside. The coordinator, the dashboard and Mongo
all bind to `127.0.0.1` — one public door, one certificate.

---

## Step by step

### 1. Push

```bash
git push
```

Watch the **images** workflow at
`https://github.com/Svaaan/HelloWorldAi/actions`. It builds the `server` target
and pushes two tags: `latest`, and `sha-<commit>`.

Roughly six minutes cold, under one when only source changed — the dependency
layer is cached and moves only when `requirements.txt` does.

The **tests** workflow runs alongside it. It does not gate the image build, so
check it went green too.

### 2. Deploy

```bash
DOMAIN=artificialintelligentduck.duckdns.org \
HOST=95.216.190.197 \
KEY=~/.ssh/hetzner \
./deploy.sh
```

It rsyncs the source (excluding `env/.env.*`, `.git`, `docker/data` and the
virtualenvs), pulls the image, and restarts.

**`DOMAIN` is needed every time.** Caddy reads it at container start, so
leaving it off means the certificate is issued for `localhost` and browsers
refuse it.

### 3. Check

```bash
curl -sI https://artificialintelligentduck.duckdns.org/ | head -1
```

`HTTP/2 200` is what you want. Then:

```bash
ssh -i ~/.ssh/hetzner ubuntu@95.216.190.197 \
  "cd HelloWorldAi && docker compose -p helloworldai -f docker/docker-compose.yml ps"
```

All four up, three of them `(healthy)`. Caddy has no healthcheck; if the site
answers, it is fine.

---

## Startup order, and why it matters

Compose waits for **health**, not for a container to exist:

```
mongo_prod   healthy  ->  coordinator  healthy  ->  dashboard  healthy  ->  caddy
```

The previous deployment of this project sat `Up` for twelve months with no
database container at all, and `docker ps` looked perfectly happy about it.
That is what the healthchecks are for. If a deploy hangs at `Waiting`, the
service below it is failing its check — look at that one's logs, not the one
that appears stuck.

---

## When something is wrong

**See what happened**

```bash
ssh -i ~/.ssh/hetzner ubuntu@95.216.190.197 \
  "docker logs helloworldai-coordinator-1 --tail 50"
```

Swap in `helloworldai-dashboard-1`, `caddy` or `mongo_prod`.

**The registry is down, or you are deploying a branch CI has not built**

```bash
BUILD_ON_SERVER=1 DOMAIN=... HOST=... KEY=... ./deploy.sh
```

Builds from the rsynced source on the box. Slower, and it needs build tooling
in production, which is exactly what the published image avoids — so treat it
as a fallback, not a habit.

**Go back to a known-good version**

Every build is tagged with its commit:

```bash
SERVER_IMAGE=ghcr.io/svaaan/helloworldai-server:sha-<commit> \
DOMAIN=... HOST=... KEY=... ./deploy.sh
```

That is the reason for the sha tags — rolling back to `latest` from yesterday
is not a thing you can do.

**Certificate trouble**

Caddy needs port 80 reachable to answer the challenge, even though everything
redirects to 443. Check the Hetzner firewall allows 80 from anywhere, and that
the domain still resolves to the server:

```bash
nslookup artificialintelligentduck.duckdns.org
```

DuckDNS records the IP of whoever loads its page, so it can quietly point at
your home connection again. It should be `95.216.190.197`.

**Nothing changed after a deploy**

Almost always the image. See *The one rule*.

---

## Things that are not obvious

**"Could not reach http://node:9100" is not a fault.** There is no node agent
on the central server and there cannot be one — a graphics card is offered from
the machine it is in. The front door hides node registration where no agent is
present, and the API answers 503 with an explanation rather than a DNS error.

**`env/.env.production` lives only on the server.** `deploy.sh` excludes
`env/.env.*` on purpose: a local copy travelling up would overwrite the
server's secrets, including an artifact encryption key that cannot be
recovered. To change a setting, edit it on the box and restart.

**Back that file up.** It holds `NODE_TOKEN_SECRET` and
`ARTIFACT_ENCRYPTION_KEY`. The database is unreadable without the second one
and there is no recovery path.

```bash
scp -i ~/.ssh/hetzner \
  ubuntu@95.216.190.197:HelloWorldAi/env/.env.production ./env-production-backup.txt
```

Keep it somewhere other than the server, alongside a database dump — one is
useless without the other.

**Only the server image is published.** The node image builds on `nvidia/cuda`
and comes to about 12.5 GB; a GitHub-hosted runner has roughly 14 GB free.
Contributors build it once when they start their node.

---

## Looking at the database

Mongo is on the loopback address with no authentication, so reach it through a
tunnel rather than publishing it:

```bash
ssh -i ~/.ssh/hetzner -L 27019:localhost:27019 ubuntu@95.216.190.197
```

Then point Compass at `mongodb://localhost:27019`. The production database is
`NodeDbProd`; development is `NodeDbDev` on port 27018 locally.

---

## First-time setup on a new server

Only needed when starting a fresh box.

```bash
# as root
apt-get update && apt-get install -y docker.io docker-compose-v2 git
adduser --disabled-password --gecos "" ubuntu
usermod -aG docker ubuntu
mkdir -p /home/ubuntu/.ssh
cp ~/.ssh/authorized_keys /home/ubuntu/.ssh/
chown -R ubuntu:ubuntu /home/ubuntu/.ssh
chmod 700 /home/ubuntu/.ssh && chmod 600 /home/ubuntu/.ssh/authorized_keys

# as ubuntu
git clone https://github.com/Svaaan/HelloWorldAi.git
cd HelloWorldAi && ./make-production-env.sh     # generates the secrets, mode 600
```

Firewall: inbound **22**, **80**, **443**. Nothing else.

DNS: an A record for your hostname pointing at the server, resolving before you
deploy — Let's Encrypt checks it.

Then deploy as above.
