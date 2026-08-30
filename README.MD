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

You want four things listed — `dashboard`, `coordinator`, `node`, `mongo_dev` —
all saying `running`. If something says `exited`, jump to
[Troubleshooting](#troubleshooting).

### Step 5 — Open it

Go to **http://localhost:3000** in your browser.

### Step 6 — Register your node

Everything happens on the front door. There are two cards — **I have data** and
**I have a GPU** — and each offers the same two things: make the key that is
your account, or load the one you already made. Registering a graphics card
used to be a second page; it is a dialog on this one now, and the GPU check
only runs when you ask to register, so somebody arriving with a dataset and a
laptop never meets it.

1. Click **Create key file** on the **I have a GPU** card
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
2. Click a node, or **Send to any node** to let the coordinator pick a free one
3. Attach your training data (see below) and fill in the form
4. Click **Send job**

The job goes into a queue. A node picks it up within about 10 seconds, trains,
and sends the result back. It appears in **your workspace**, where you can
download the trained model. Watch it happen:

```bash
docker compose -f docker/docker-compose.test.yml logs -f node
```

Without a dataset attached the node trains on made-up numbers, which only proves
the plumbing works. Results are marked `synthetic_data: true` so you can tell
real work from a dry run.

### What you can train on

Two kinds of file, and the form follows whichever you attach:

| You upload | You get | What the file looks like |
|---|---|---|
| `.csv` | A classifier | Every column a feature except the last, which is the label. Labels may be numbers or names. |
| `.txt` | A language model | Any plain text. It learns to continue it, one character at a time. |

Everything about the model's *shape* is read from your file — how many inputs,
how many classes, how long the sequences are. You choose its *size*: how wide,
how deep, how long to train.

**How much data you need.** Text is the hungry one. Measured on this service,
same hardware, same architecture family:

| Corpus | Held-back accuracy | What it produced |
|---|---|---|
| 71 KB | 41% | Word-shaped nonsense |
| 310 KB | 62% | Syntactically plausible lines |
| 25 MB | 77% | Correctly spelled words in sentences |

All passed verification; all were real training. Around 1 MB is where
sentences start to hold together, and the limit on a text upload is 128 MB. The upload says so before you send the job,
and the result is graded **barely learned / learned / learned well** alongside
the Verified badge — because "Verified" only means the model is genuine, not
that it is good.

## Getting the model back

The download is a zip laid out the way models normally are:

```
model.safetensors    the weights
config.json          the shape, what the inputs are called, what the answers mean
tokenizer.json       how text becomes token ids   (language models only)
load_model.py        rebuilds it and runs it, needing nothing from this project
README.md            the two commands
```

```bash
unzip my-model.zip && cd my-model
python load_model.py model.safetensors
python load_model.py model.safetensors --prompt "Once upon a"
```

safetensors rather than `.pt` for the same reason this project never used
`torch.save`: loading a pickle executes whatever is in it, and these files are
meant to be passed between people. To run the model outside Python at all,
convert it once — `--export my-model.onnx`.

**If you uploaded a spreadsheet you may not want a model file at all.** Send a
CSV of new rows from the workspace and it comes back as the same file with a
predicted column and a confidence added — no download, no Python. Columns are
matched by name, so their order does not matter and extra columns are left
alone; you can send back the very file you trained on and see the answer next
to the truth.

The `.npz` is this project's own arrangement of arrays — safe to load, and not
a format anything else reads. To use the model somewhere else, convert it:

```bash
python load_model.py my-model.npz --export my-model.pt
```

`.pt` is TorchScript: it loads with `torch.jit.load` and needs no model code
beside it. `.onnx` also works and runs almost anywhere without Python, though
writing it needs one more package — the script names which one, because it
differs between versions of torch.

### Where the GPU comes in

Only for the training loop itself. Everything before it is ordinary CPU work you
do on your own machine:

```
your files  →  turn them into numbers  →  [ TRAIN ]  →  a model file
.txt .csv       CPU, on your machine       ← GPU →       you download
.jpg .mp3       decode, tokenise, extract   the network
```

That is why the uploads above are CSV and text: they are the two formats this
service can turn into numbers for you. Anything else — images, audio, Parquet,
annotations — has to be converted first. If you can turn your data into a table
of numbers with a label per row, save it as CSV and it will train. If you can
turn it into token ids yourself, you can upload a packed `.npz` of `x` and `y`
arrays directly to `POST /artifacts` and skip the conversion entirely.

What is **not** supported, on purpose: uploading a `.py` training script. Running
a stranger's code on a contributor's machine is the exact thing this project is
built to avoid. The form is the configuration file.

---

# Part 3 — Let a friend donate their GPU

Your friend does **not** clone anything or read any of this. They run one line.

They need: Windows with an NVIDIA GPU, WSL 2 installed, and the NVIDIA driver
installed **on Windows**.

Inside their Ubuntu (WSL) terminal:

```bash
bash <(curl -fsSL http://YOUR-SERVER:3000/static/scripts/setup-node.sh) --coordinator http://YOUR-SERVER:3000
```

Replace `YOUR-SERVER` with wherever your site is reachable — one address,
not two. Everything the node agent needs is forwarded through the dashboard, so
the coordinator's own port never has to be exposed. The **Setup guide** page on
your site generates this line with the address filled in already; send them that
page.

The script installs Docker and the GPU toolkit, checks a container can really
see their GPUs, downloads the node, and starts it. It is safe to run twice.

Then they open **http://localhost:3000** on their own machine and register,
exactly like Step 6 above — the **I have a GPU** card, **Create key file**.

---

# Part 4 — Put it on a server

Parts 1 to 3 run everything on your own machine. This part is about running the
coordinator centrally, once, so contributors and people with data can reach it
without you being switched on.

Contributors do **not** follow this. They run the one-liner from Part 3.

### What you need

- A Linux server with a public address and a domain pointing at it.
- Docker with Compose v2.
- A TLS certificate. This is not optional -- see below.

No GPU is needed on this machine. The coordinator hands work out; it never
trains anything itself.

### Step 1 — Settings

```bash
git clone https://github.com/Svaaan/HelloWorldAi.git
cd HelloWorldAi
cp env/.env.example env/.env.production
```

Open `env/.env.production` and generate the two secrets it asks for:

```bash
python3 -c "import secrets;print(secrets.token_urlsafe(32))"                 # NODE_TOKEN_SECRET
python3 -c "import base64,os;print(base64.b64encode(os.urandom(32)).decode())" # ARTIFACT_ENCRYPTION_KEY
```

`NODE_TOKEN_SECRET` signs the tokens nodes use to prove who they are. Leave it
empty and a throwaway key is made at startup, which logs every node out every
time you restart.

`ARTIFACT_ENCRYPTION_KEY` is **optional**, and the service runs either way.

Set it, and submitted datasets are encrypted before they are written to the
database; the send-work page then tells submitters so. Leave it empty and they
are stored in the clear, the coordinator says so in its log at startup, and the
page tells submitters that instead. Nothing breaks without it -- what changes is
what you can honestly claim.

If you do set one, back it up somewhere other than the server, alongside the
database. Encrypted datasets cannot be read without it and there is no recovery
path. If you have never set one, there is nothing to recover and nothing to
look for: an empty value is a working configuration, not a missing key.

The file is gitignored. Keep it that way.

### Step 2 — Start it

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

That runs three containers: the coordinator, the public dashboard, and a
MongoDB with a named volume so registrations and job history survive a restart.

Check it came up:

```bash
docker compose -f docker/docker-compose.yml ps
curl -s localhost:3000/available-nodes
```

### Step 3 — Put TLS in front of it

Both services deliberately bind to `127.0.0.1`, so nothing is reachable from
the internet until you place a reverse proxy in front of them.

This matters more than it used to. Node session tokens and submitter keys now
travel in request headers on every call. Over plain HTTP anyone on the path can
read them, and a stolen submitter key hands over that person's jobs and their
trained models. Parts of the interface also quietly stop working without a
secure context -- the copy buttons on the setup page, for one.

With Caddy, the whole configuration is:

```
your-domain.com {
    reverse_proxy 127.0.0.1:3000
}

coordinator.your-domain.com {
    reverse_proxy 127.0.0.1:8100
}
```

nginx with certbot works equally well; the requirement is TLS, not a particular
proxy. If you genuinely mean to serve plain HTTP, set `BIND_ADDRESS=0.0.0.0`
and accept what that means.

### Step 4 — Send contributors to your site

Point people at `https://your-domain.com/setup`. The install command on that
page fills in your address automatically, so there is nothing for them to edit.

### Keeping it running

```bash
docker compose -f docker/docker-compose.yml logs -f coordinator
docker compose -f docker/docker-compose.yml restart coordinator
docker compose -f docker/docker-compose.yml down          # keeps the database
```

### Where the images come from

The coordinator and the dashboard run a published image rather than one built
on the server. CI builds it on every push to `main` and pushes it to
`ghcr.io/svaaan/helloworldai-server`, tagged `latest` and with the commit sha,
so a deployment can be traced to the code in it and rolled back to a specific
one rather than to whatever `latest` was yesterday.

`deploy.sh` pulls it. Building on the box made the smallest machine in the
system do the most expensive job, needed build tooling in production, and left
the stack down when a build failed.

To pin a particular build:

```bash
SERVER_IMAGE=ghcr.io/svaaan/helloworldai-server:sha-<commit> ./deploy.sh
```

To build on the server anyway -- no registry access, or a branch CI has not
built:

```bash
BUILD_ON_SERVER=1 ./deploy.sh
```

The node image is not published. It builds on `nvidia/cuda` and comes to about
12.5 GB, which does not fit a GitHub-hosted runner, and a contributor builds it
once when they start their node.

### Looking at the data

Both databases are published on the loopback address so a GUI client can
attach. They are on different ports, so development and production can run side
by side without one pretending to be the other.

| Stack | Compass connection string | Database |
|---|---|---|
| development (`docker-compose.test.yml`) | `mongodb://localhost:27018` | `NodeDbDev` |
| production (`docker-compose.yml`) | `mongodb://localhost:27019` | `NodeDbProd` |

Neither instance has authentication, so neither is bound to anything but
`127.0.0.1`. To read a production database on a remote server, forward the port
rather than publishing it:

```bash
ssh -L 27019:localhost:27019 user@server
```

Compass then connects to `mongodb://localhost:27019` as though it were local.

The database name comes from `MONGO_DB` and the server from `MONGO_URI`, so an
environment is a matter of configuration. It used to be the literal string
`NodeDbTest` in the source, which meant a production deployment would have
written its real registrations and job history into a database called Test.

---

Back up the `mongo-data` volume and `env/.env.production` together. The
database is useless without the encryption key.

---

## Running the tests

No Docker needed, but the tests do need their own packages -- pytest above all,
which the main `requirements.txt` does not list. Once per machine:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

On macOS or Linux that second path is `.venv/bin/python`. Then, from the
project folder:

```bash
.venv/Scripts/python -m pytest tests/ -q
```

That should say `409 passed`. A single file, when you are working on one:

```bash
.venv/Scripts/python -m pytest tests/test_gpu_match.py -q
```

Run them through pytest rather than as plain scripts. Ten of these files have
no `__main__` block, so `python tests/test_scoring.py` exits successfully
having run nothing at all -- which looks exactly like passing.

Watch the count for the same reason. Tests needing PyTorch, ONNX Runtime or
NumPy skip themselves rather than fail, so an incomplete environment reports
a cheerful `passed` with about fifty tests quietly missing. `requirements-dev.txt`
installs all of them; if the number is short of 409, something did not install.

| File | What it checks |
|---|---|
| `test_gpu_match.py` | Identifying a GPU model correctly |
| `test_pool_planner.py` | Splitting work across mismatched GPUs |
| `test_thermal_policy.py` | Stopping before a contributor's card cooks itself |
| `test_artifacts.py` | That a malicious file cannot run code on your machine |
| `test_artifact_crypto.py` | Datasets being unreadable in the database |
| `test_artifact_access.py` | Who may download a dataset, holdout or model |
| `test_trainer.py` | That training actually works and is mathematically sound |
| `test_task_executor.py` | Job dispatch |
| `test_job_spec.py` | Refusing a broken job before it reaches a GPU |
| `test_job_control.py` | Stopping a running job cleanly |
| `test_node_picker.py` | Choosing which machine gets the work |
| `test_redispatch.py` | Moving a declined job to someone else |
| `test_model_manifest.py` | That a downloaded model can be rebuilt and run |
| `test_submitter.py` | Identity for the person sending work |
| `test_verification.py` | Catching a contributor who fakes results |
| `test_dataset_growth.py` | Adding more files to a dataset, and merging their labels |
| `test_sampling.py` | Asking a finished language model to write something |
| `test_warm_start.py` | Continuing a run from a trained model, and the fraud that invites |
| `test_export.py` | Converting a downloaded model to ONNX or TorchScript |
| `test_model_bundle.py` | Packaging the finished model as weights, config and loader |
| `test_scoring.py` | Answering a spreadsheet with a spreadsheet |
| `test_node_reachability.py` | That everything a node agent calls is reachable from one address |
| `test_text_dataset.py` | Turning a text file into something a language model can train on |
| `test_advice.py` | Warning that data is too small, and grading how well a model learned |
| `test_retry.py` | Re-running a job on the same data with different settings |

The rules deciding which side of the network sees which page -- and which
navigation belongs on it -- live in the browser, so they are checked with Node
instead:

```bash
node tests/test_role_pages.js
```

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

Working, and covered by **409 tests**:

- Detecting GPUs and measuring what they can really do
- Splitting one batch across mismatched GPUs, proportional to real speed
- Actual PyTorch training, single or multi GPU
- Safe file format for data and weights — a hostile file cannot execute code
- Catching contributors who return fabricated results
- Job queue with retries for machines that vanish mid-job
- Stopping a job on the submitter's word, and running it again
- Choosing which machine gets a job, and moving it on if that machine declines
- An identity for the person sending work, and a way to collect the model
- Datasets encrypted in the database, and deleted once a job is done
- Training a language model on a text file you upload, and generating text
  from what comes back

Built but **not yet run against a live server**, so expect rough edges:

- The installer script (written and checked, never run on a fresh machine)
- Everything in Part 4 — the central deployment has not been exercised end to
  end on real hardware behind a real domain

Not built yet:

- **Payments.** Nobody is paid for anything yet. The interface no longer
  implies otherwise: there is no price on a node and nothing claims a
  contributor is earning.
- **Splitting one job across several people's machines.** Home internet is far
  too slow for that. One job runs on one person's machine, using all of that
  person's GPUs.
- **Password recovery of any kind.** Your identity is a key file. Both pages
  warn you while that key exists only in your browser. Lose it and
  the jobs and models it owns are gone.

### What privacy you actually get

Worth being exact about, because it is easy to assume more than is true.

The contributor running your job **can read your training data**. Training
needs the numbers in the clear on their GPU, and the techniques that would
avoid that are far too slow or need hardware consumer cards do not have. Do not
send anything you would not hand to a stranger.

What is true:

- Column names never leave your browser — the node receives unlabelled numbers.
- Datasets are encrypted in the database, if you set an encryption key.
- The node keeps data in memory only and never writes it to disk.
- Both copies are deleted once the job has finished and been checked.
- Part of your data is held back from the node and used to check the result.

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
