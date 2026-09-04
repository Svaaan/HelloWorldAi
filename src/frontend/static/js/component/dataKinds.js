// What a file has to look like, written once.
//
// Two pages ask somebody for training data: the send-work form, where it is
// the whole point, and the node page's self-test, where a contributor can put
// their own CSV through the same training code to see what their card does
// with real data.
//
// Both explained the same rule -- every column is a feature except the last,
// which is the label -- in their own words and their own markup. Two copies of
// one fact, and the sort that drifts: change the convention and one of them
// goes on describing the old one, on the page where somebody is least likely
// to have the other open beside it.
//
// The forms themselves are not the same and should not be. The sender accepts
// several files and either kind, because the choice of file is the choice of
// model; the self-test takes one CSV, because it is measuring a graphics card
// rather than training something to keep. What they share is the explanation.

const KINDS = {
  csv: {
    extension: ".csv",
    title: "Rows of numbers",
    detail: "Every column is a feature except the last, which is the label. "
          + "You get a classifier.",
  },
  text: {
    extension: ".txt",
    title: "Any plain text",
    detail: "Books, notes, transcripts, code. You get a model that continues "
          + "text.",
  },
};

/** The file extensions an input should accept, for the kinds named. */
export function acceptFor(kinds) {
  const map = {
    csv: ".csv,text/csv",
    text: ".txt,.md,text/plain",
  };
  return kinds.map((k) => map[k]).join(",");
}

/**
 * A list explaining each kind of file, ready to append.
 *
 * @param {string[]} kinds  which of them apply here -- ["csv"] on the node
 *   page, ["csv", "text"] on the send-work form.
 */
export function dataKindsList(kinds = ["csv", "text"]) {
  const list = document.createElement("ul");
  list.className = "data-kinds";

  for (const name of kinds) {
    const kind = KINDS[name];
    if (!kind) continue;

    const item = document.createElement("li");

    const ext = document.createElement("code");
    ext.className = "data-kind-ext";
    ext.textContent = kind.extension;
    item.appendChild(ext);

    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = kind.title;
    body.appendChild(title);

    const detail = document.createElement("span");
    detail.textContent = kind.detail;
    body.appendChild(detail);

    item.appendChild(body);
    list.appendChild(item);
  }

  return list;
}
