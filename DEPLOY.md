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

**Only the server image is published.** The node image builds on `nvidia/cuda`
and comes to about 12.5 GB; a GitHub-hosted runner has roughly 14 GB free.
Contributors build it once when they start their node.

**There are three requirements files, one per image.** `requirements.txt` for
the server, `requirements-node.txt` for the node agent, and
`requirements-dashboard.txt` for the dashboard a contributor runs beside their
node — which is a proxy, so it needs five packages and 260 MB rather than the
1.5 GB it used to carry. `tests/test_image_requirements.py` checks each file
against the imports its entry point actually reaches, because the way this goes
wrong is an image that builds and then will not start.

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

---

## Backups

Everything worth keeping is on one machine: one server, one MongoDB, one copy
of a key that cannot be reissued. Run this from your own machine, not the
server — a backup that lives on the thing it is backing up is not one.

```bash
HOST=95.216.190.197 KEY=~/.ssh/hetzner bash ./backup.sh --verify
```

It writes a dated directory to `~/helloworldai-backups` holding two files:

| | |
|---|---|
| `database.archive.gz` | every collection, GridFS included — the datasets people submitted and the models that came back |
| `env.production` | the server's secrets |

**Neither is any use alone.** Submitted datasets are encrypted with
`ARTIFACT_ENCRYPTION_KEY`, which exists only in that env file. Restore the
database without it and you have a list of jobs and a pile of bytes nobody can
read. That is also why the directory is written mode 700 and why the script
refuses to write anywhere inside this repository: it holds the production
secrets, and a backup in a git working tree is one `git add -A` from being
published.

`--verify` restores the archive into a throwaway MongoDB and counts what comes
out. Use it. `mongodump` exits 0 for an empty database exactly as happily as
for a full one, so the failure this catches is the silent one — a backup that
has been running nightly and capturing nothing.

### Getting it back

On a rebuilt server, with the stack up:

```bash
scp -i ~/.ssh/hetzner database.archive.gz ubuntu@95.216.190.197:/tmp/
ssh -i ~/.ssh/hetzner ubuntu@95.216.190.197 \
  "docker exec -i mongo_prod mongorestore --archive --gzip --drop < /tmp/database.archive.gz"
```

`--drop` replaces the collections in the archive. That is what you want when
recovering and emphatically not what you want on a database with newer data in
it — take a fresh backup first if there is any doubt.

Then put the secrets back, or the restored datasets stay unreadable:

```bash
scp -i ~/.ssh/hetzner env.production ubuntu@95.216.190.197:HelloWorldAi/env/.env.production
ssh -i ~/.ssh/hetzner ubuntu@95.216.190.197 \
  "cd HelloWorldAi/docker && docker compose -p helloworldai -f docker-compose.yml restart"
```

### Doing it without being asked

The script is deliberately run from your machine, so it does not schedule
itself. On Linux or WSL, a nightly cron entry:

```bash
0 3 * * * HOST=95.216.190.197 KEY=$HOME/.ssh/hetzner bash $HOME/HelloWorldAi/backup.sh >> $HOME/backup.log 2>&1
```

Prune old ones yourself; the script never deletes anything, because the failure
mode of over-eager cleanup is worse than a full disk.

---

## The smoke test

```bash
docker compose -p hwai-smoke -f docker/docker-compose.smoke.yml \
  up --build --abort-on-container-exit --exit-code-from smoke
```

Brings up MongoDB, the coordinator, the production dashboard and the
contributor dashboard from the images that ship, and runs one job through them:
register a node, prove its key, upload a dataset, submit, claim, train, report,
verify, download the model. Exit code 0 means all of that worked.

Two things it does that the unit suites cannot:

**It uses the built images, with no source mounted.** `docker-compose.test.yml`
bind-mounts `../src`, which is right for development and means it never tests
what was put in an image. The dashboard that crash-looped on `No module named
'bson'` built perfectly.

**It goes through the dashboard, not the coordinator.** The proxy is where a
path reaches the wrong service without anything saying so — `/connect-node`
meant two different things for months, and no contributor's node could register
through the public address at all.

There is no node agent in it. That image is built on `nvidia/cuda` and wants a
graphics card CI does not have, so `tests/smoke/smoke_run.py` speaks the node's
half of the protocol and calls the same training executor the real agent calls.
Everything on the other side of those calls is real.

It runs in CI on every push, as the `smoke` job, and takes a few minutes
because it builds the images.
