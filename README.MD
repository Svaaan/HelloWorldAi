# HelloWorldAi

Pool ordinary people's GPUs into shared compute, so someone with data but no
hardware can train a model on someone else's idle graphics card.

> This README is in English because the code and comments are. Ask and I'll
> translate it to Swedish.

---

## What this actually is

Three kinds of people, three different things to run:

| Who | What they have | What they run |
|---|---|---|
| **You (the operator)** | A server | The **coordinator** — the switchboard. Keeps the node registry and the job queue. |
| **A contributor** | A PC with NVIDIA GPUs | The **node agent** — does the actual training. Runs on their machine, at home. |
| **A submitter** | Data, no GPU | Nothing. They use the website to upload data and collect results. |

The important bit: **the coordinator never connects to a contributor's PC.**
Home routers block incoming connections. Instead the contributor's node reaches
*out* and asks "any work for me?". That's why it works from a bedroom.

A contributor with several different GPUs — say a 3070, a 3050 and an old 1660 —
gets all of them used at once, with each card given a share of the work sized to
how fast it actually is.

---

## Before you start

Install these:

1. **Docker Desktop** — https://www.docker.com/products/docker-desktop/
   (Windows users: it will ask to enable WSL 2. Say yes.)
2. **Git** — https://git-scm.com/downloads

Only if you want to *use* a GPU (not needed just to try the software):

3. **NVIDIA driver** — https://www.nvidia.com/Download/index.aspx
   On Windows install it on **Windows itself**, not inside WSL.
4. **NVIDIA Container Toolkit** — this is what lets Docker see your GPU.
   The setup script in Part 3 installs it for you.

Check Docker works before going further. This should print a version:

```bash
docker --version
```

---

# Part 1 — Run the whole thing on your own machine

This gets everything running locally so you can click around. About 10 minutes,
most of it waiting for the first build.

### Step 1 — Get the code

```bash
git clone https://github.com/Svaaan/HelloWorldAi.git
```

Then move into the folder:

```bash
cd HelloWorldAi
```

### Step 2 — Create the settings file

**Do not skip this.** The settings files are deliberately not stored in Git (they
can hold secrets), so a fresh copy of the project has none. Without this step
Docker stops immediately with `env file ... not found`.

On **Windows (PowerShell)**:

```powershell
$secret = python -c "import secrets;print(secrets.token_urlsafe(32))"
@(
  "ENV=test"
  "USE_DOCKER=true"
  "COORDINATOR_URL=http://coordinator:8100"
  "NODE_URL=http://node:9100"
  "DASHBOARD_PORT=3000"
  "COORDINATOR_PORT=8100"
  "NODE_PORT=9100"
  "NODE_TOKEN_SECRET=$secret"
) | Set-Content -Path env/.env.test -Encoding utf8
```

Paste the whole block at once, not line by line.

On **Mac or Linux**:

```bash
printf 'ENV=test\nUSE_DOCKER=true\nCOORDINATOR_URL=http://coordinator:8100\nNODE_URL=http://node:9100\nDASHBOARD_PORT=3000\nCOORDINATOR_PORT=8100\nNODE_PORT=9100\nNODE_TOKEN_SECRET=%s\n' "$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" > env/.env.test
```

What that file means, in plain words:

| Line | Meaning |
|---|---|
| `COORDINATOR_URL` | Where the switchboard lives |
| `NODE_URL` | Where the worker lives |
| `NODE_TOKEN_SECRET` | A password the coordinator uses to sign login tickets for nodes. Random is correct. |

If you skip the secret, everything still runs, but every node gets logged out
whenever the coordinator restarts.

### Step 3 — Start it

```bash
docker compose -f docker/docker-compose.test.yml up -d --build
```

**The first run takes 5–15 minutes.** It downloads a large NVIDIA base image and
installs Python packages. Later runs take seconds. `-d` means "run in the
background".

**Have an NVIDIA GPU and want it used?** Add the GPU overlay instead:

```bash
docker compose -f docker/docker-compose.test.yml -f docker/docker-compose.gpu.yml up -d --build
```

### Step 4 — Check that it started

```bash
docker compose -f docker/docker-compose.test.yml ps
```

You want four things listed — `dashboard`, `coordinator`, `node`, `mongo_test` —
all saying `running`. If something says `exited`, jump to
[Troubleshooting](#troubleshooting).

### Step 5 — Open it

Go to **http://localhost:3000** in your browser.

### Step 6 — Register your node

1. Click **Register New Node**
2. Type any name
3. Click **Confirm Registration**
4. **Download the key file and keep it.** It is the only proof the node is yours.
   Lose it and you cannot reconnect that node — there is no password reset.

> **No NVIDIA GPU?** Registration will be refused with "No valid GPU detected".
> That is intended — a node with nothing to contribute should not join. You can
> still browse the site and run the tests.

### Step 7 — Stop it when you're done

```bash
docker compose -f docker/docker-compose.test.yml down
```

Add `-v` to also wipe the database and start clean next time.

---

# Part 2 — Send a training job

Once a node shows as **Available** on the dashboard:

1. Go to **http://localhost:3000/distribution**
2. Click your node in the list
3. Paste a job description into the box, for example:

```json
{
  "task_type": "llm_training",
  "model_name": "my-first-model",
  "model_spec": { "architecture": "mlp", "hidden_dim": 64, "depth": 2 },
  "hyperparameters": { "steps": 50, "batch_size": 64, "learning_rate": 0.01 }
}
```

What each part means:

| Field | Meaning |
|---|---|
| `task_type` | The only kind currently supported is `llm_training` |
| `model_spec.architecture` | `mlp` for ordinary tables of numbers, `transformer` for text |
| `hyperparameters.steps` | How many rounds of training |
| `hyperparameters.batch_size` | How many examples per round — this is what gets split across the GPUs |

4. Click **Send Task Request**

The job goes into a queue. The node picks it up within about 10 seconds, trains,
and sends the result back. Watch it happen:

```bash
docker compose -f docker/docker-compose.test.yml logs -f node
```

Without a dataset attached the node trains on made-up numbers, which only proves
the plumbing works. Results are marked `synthetic_data: true` so you can tell
real work from a dry run.

---

# Part 3 — Let a friend donate their GPU

Your friend does **not** clone anything or read any of this. They run one line.

They need: Windows with an NVIDIA GPU, WSL 2 installed, and the NVIDIA driver
installed **on Windows**.

Inside their Ubuntu (WSL) terminal:

```bash
bash <(curl -fsSL http://YOUR-SERVER:3000/static/scripts/setup-node.sh) --coordinator http://YOUR-SERVER:8100
```

Replace `YOUR-SERVER` with wherever your coordinator is reachable. The
**Get Started** page on your site generates this line with the right address
filled in already — send them that page.

The script installs Docker and the GPU toolkit, checks a container can really
see their GPUs, downloads the node, and starts it. It is safe to run twice.

Then they open **http://localhost:3000** on their own machine and register,
exactly like Step 6 above.

---

## Running the tests

No Docker needed. From the project folder:

```bash
python tests/test_gpu_match.py
```

Each test file runs on its own:

| File | What it checks |
|---|---|
| `test_gpu_match.py` | Identifying a GPU model correctly |
| `test_pool_planner.py` | Splitting work across mismatched GPUs |
| `test_artifacts.py` | That a malicious file cannot run code on your machine |
| `test_trainer.py` | That training actually works and is mathematically sound |
| `test_task_executor.py` | Job dispatch |
| `test_verification.py` | Catching a contributor who fakes results |

Tests needing PyTorch or NumPy skip themselves politely if you don't have them,
so nothing here breaks on a plain Python install.

---

## Troubleshooting

**`env file /...env.test not found`**
You skipped Step 2. Go back and create the settings file.

**`could not select device driver "nvidia" with capabilities: [[gpu]]`**
You used the GPU overlay on a machine without a working NVIDIA setup. Drop
`-f docker/docker-compose.gpu.yml` and run without it.

**Registration says "No valid GPU detected"**
Either the machine has no NVIDIA GPU, or Docker cannot see it. Test with:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

If that fails, the NVIDIA Container Toolkit is missing or misconfigured. If it
works but registration still fails, you started without the GPU overlay.

**A container says `exited`**
Read why:

```bash
docker compose -f docker/docker-compose.test.yml logs coordinator
```

**Node shows as Disconnected after a few minutes**
The node reports in every 60 seconds and is marked dead after 5 minutes of
silence. Check the node's logs for rejected session tokens.

**Port 3000 already in use**
Something else is using it. Stop that program, or change `DASHBOARD_PORT` in
your settings file and the port mapping in the compose file.

---

## Honest status

Working and covered by tests (**111 tests**):

- Detecting GPUs and measuring what they can really do
- Splitting one batch across mismatched GPUs, proportional to real speed
- Actual PyTorch training, single or multi GPU
- Safe file format for data and weights — a hostile file cannot execute code
- Catching contributors who return fabricated results
- Job queue with retries for machines that vanish mid-job

Built but **not yet run against a live server**, so expect rough edges:

- The installer script (written and checked, never run on a real machine)
- Uploading and downloading datasets through the coordinator
- Automatic verification when a result arrives

Not built yet:

- **User accounts.** Your identity is the key file you downloaded, nothing more.
- **Choosing a node for you.** You pick one by hand from the list.
- **Payments.** The price shown in the interface is decoration.
- **Privacy.** Uploaded data is stored unencrypted and lands on a stranger's
  computer. Do not upload anything sensitive.
- **Splitting one job across several people's machines.** Home internet is far
  too slow for that. One job runs on one person's machine, using all of that
  person's GPUs.

---

## Where things live

```
src/
  app.py                      Starts the website
  backend/
    coordinator.py            The switchboard: registry, job queue, verification
    node.py                   The worker: asks for jobs, runs them, reports back
    proxypage.py              Passes browser requests to the right service
    service/
      gpuBenchmark.py         Measures what each GPU can really do
      poolPlanner.py          Divides work across mismatched GPUs
      trainer.py              The actual training
      taskExecutor.py         Picks which kind of job to run
      artifacts.py            Safe file format for anything sent over the network
      verification.py         Detects faked results
      gpuMatch.py             Works out which GPU model you have
      tokenService.py         Signs login tickets for nodes
  frontend/                   The website itself
docker/
  docker-compose.test.yml     Everything on one machine (start here)
  docker-compose.node.yml     For a contributor donating their GPU
  docker-compose.gpu.yml      Add-on that hands your GPU to Docker
tests/                        Run any file directly with python
```

---

## Useful commands

```bash
docker compose -f docker/docker-compose.test.yml logs -f node
```

```bash
docker compose -f docker/docker-compose.test.yml restart node
```

```bash
docker compose -f docker/docker-compose.test.yml down -v
```

(The last one deletes the database too — a completely fresh start.)
