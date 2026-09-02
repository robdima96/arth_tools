# -*- coding: utf-8 -*-
"""Streamlit library: anatomy → modality → task → tool, then Infer / Train / Info."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from arth_tools.data.preview import PreviewRow, PREVIEW_N, sample_preview_rows
from arth_tools.ui.catalog import ToolEntry, load_catalog


def _architectures() -> list[str]:
    import arth_tools.training.backbones  # noqa: F401 — register defaults

    from arth_tools.models.registry import registered_architectures

    return registered_architectures()


def _citation_line(cite: Any) -> str:
    if not isinstance(cite, dict):
        return str(cite)
    bits = [
        cite.get("authors"),
        cite.get("title"),
        str(cite["year"]) if cite.get("year") else None,
        cite.get("venue"),
    ]
    text = ". ".join(str(b) for b in bits if b)
    doi = cite.get("doi")
    url = cite.get("url")
    code_url = cite.get("code_url")
    if doi:
        text = f"{text} doi:{doi}" if text else f"doi:{doi}"
    elif url:
        text = f"{text} {url}" if text else str(url)
    if code_url:
        text = f"{text} code:{code_url}" if text else str(code_url)
    return text or "(empty citation)"


def render_info(tool: ToolEntry) -> None:
    info = tool.info or {}
    st.subheader("Summary")
    summary = info.get("summary") or tool.task_name
    st.write(summary)
    meta = [
        f"Anatomy: {tool.anatomy_name}",
        f"Modality: {tool.modality_name}",
    ]
    if tool.num_classes is not None:
        meta.append(f"Classes: {tool.num_classes}")
    st.caption(" · ".join(meta))
    notes = str(info.get("notes") or "").strip()
    if notes:
        st.write(notes)

    st.subheader("Citations")
    citations = info.get("citations") or []
    if not citations:
        st.write("None yet.")
    else:
        for cite in citations:
            st.markdown(f"- {_citation_line(cite)}")

    st.subheader("Training methods")
    methods = str(info.get("methods") or "").strip()
    st.write(methods if methods else "None yet.")

    st.subheader("Parameters")
    params = info.get("parameters") or {}
    if not params:
        st.write("None yet.")
    else:
        for key, value in params.items():
            st.markdown(f"- **{key}:** `{value}`")

    st.subheader("Provided models")
    models = info.get("models") or []
    if not models:
        st.write("No packaged weights yet; add later.")
    else:
        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or "model"
                extra = model.get("notes") or ""
                st.markdown(f"- **{name}** — {extra}" if extra else f"- **{name}**")
            else:
                st.markdown(f"- {model}")


def _require_dir(label: str, raw: str) -> Path | None:
    text = (raw or "").strip()
    if not text:
        st.error(f"{label} is required.")
        return None
    path = Path(text)
    if not path.is_dir():
        st.error(f"{label} is not a folder: {path}")
        return None
    return path


def _preview_state_key(kind: str, tool_id: str) -> str:
    return f"preview_{kind}_{tool_id}"


def _preview_matches(state: dict[str, Any] | None, root: Path) -> bool:
    if not state or not state.get("paths"):
        return False
    try:
        return Path(str(state["root"])).resolve() == root.resolve()
    except OSError:
        return False


def _grid_images(rows: list[PreviewRow], *, attr: str, cols: int = 5) -> None:
    for start in range(0, len(rows), cols):
        chunk = rows[start : start + cols]
        columns = st.columns(len(chunk))
        for col, row in zip(columns, chunk):
            with col:
                img = getattr(row, attr)
                if img is None:
                    st.write("—")
                    continue
                st.image(
                    img,
                    caption=f"{row.path.name} · {row.patient_id}",
                    use_container_width=True,
                )


def render_preview_grid(
    rows: list[PreviewRow],
    *,
    show_processed: bool,
    processed_size: tuple[int, int] | None = None,
) -> None:
    if not rows:
        st.warning("No readable images in this folder.")
        return
    st.subheader("Loaded (before preprocess)")
    st.caption(f"Random sample of {len(rows)} (up to {PREVIEW_N}). Photometric decode only.")
    _grid_images(rows, attr="loaded_u8")
    if show_processed:
        st.subheader("After preprocess (model input)")
        size_note = ""
        if processed_size is not None:
            size_note = f" Resized to {processed_size[0]}×{processed_size[1]}."
        st.caption(
            "Same files after the train recipe (ROI / window / resize). "
            f"Z-score is applied at fit start, not in this grid.{size_note}"
        )
        _grid_images(rows, attr="processed_u8")


def _collect_infer_items(tool: ToolEntry, root: Path) -> list[dict[str, Any]]:
    if tool.infer_backend == "classical":
        from arth_tools.hka.run import iter_dicom_paths

        return [{"path": p, "patient_id": p.parent.name or p.stem} for p in iter_dicom_paths(root)]
    from arth_tools.inference.infer import collect_image_items

    return collect_image_items(root)


def run_infer_panel(tool: ToolEntry) -> None:
    key = tool.tool_id
    if tool.infer_backend == "stub":
        st.info(
            "This tool’s published pipeline is not wired for Infer yet. "
            "See Info for the paper, license, and download link. "
            "To grade your own labelled DICOMs, pick the arth_tools CNN tool under this task."
        )
        st.button("Run inference", type="primary", disabled=True, key=f"{key}_infer_go")
        return

    dicom_raw = st.text_input("DICOM folder", key=f"{key}_infer_dicom")
    source_kind = ""
    run_dir = ""
    bundle = ""
    if tool.infer_backend == "run_or_bundle":
        source_kind = st.radio(
            "Model source",
            ("Existing run directory", "Frozen bundle"),
            key=f"{key}_infer_source",
        )
        if source_kind == "Existing run directory":
            run_dir = st.text_input("Run directory", key=f"{key}_run_dir")
        else:
            bundle = st.text_input("Bundle directory", key=f"{key}_bundle")
    elif tool.infer_backend == "classical":
        st.caption("Built-in measurement (no trained model).")

    out_csv = st.text_input(
        "Output CSV (optional)",
        key=f"{key}_infer_out",
        placeholder="leave blank for a default next to the model or output dir",
    )

    state_key = _preview_state_key("infer", key)
    if st.button("Load preview", key=f"{key}_infer_preview"):
        root = _require_dir("DICOM folder", dicom_raw)
        if root is not None:
            try:
                items = _collect_infer_items(tool, root)
                rows = sample_preview_rows(items, recipe=None, n=PREVIEW_N, seed=0)
                st.session_state[state_key] = {
                    "root": str(root.resolve()),
                    "paths": [str(r.path) for r in rows],
                }
                if not rows:
                    st.error("No readable images under that folder.")
            except Exception as exc:
                st.error(str(exc))

    preview_root = Path(dicom_raw.strip()) if dicom_raw.strip() else None
    state = st.session_state.get(state_key)
    ready = preview_root is not None and preview_root.is_dir() and _preview_matches(state, preview_root)
    if ready and state is not None:
        rows = sample_preview_rows(state["paths"], recipe=None, n=PREVIEW_N, seed=0, shuffle=False)
        render_preview_grid(rows, show_processed=False)
    else:
        st.caption("Load a preview of decoded images before running inference.")

    if st.button("Run inference", type="primary", disabled=not ready, key=f"{key}_infer_go"):
        root = _require_dir("DICOM folder", dicom_raw)
        if root is None:
            return
        try:
            if tool.infer_backend == "classical":
                from arth_tools.hka.config import load_hka_config
                from arth_tools.hka.run import run_hka

                cfg = load_hka_config(tool.yaml_path)
                cfg.dicom_root = root
                with st.spinner("Running HKA…"):
                    payload = run_hka(cfg)
                st.success(f"Wrote {payload.get('n_ok', 0)} annotated DICOMs → {payload.get('output_dir')}")
                result_rows = payload.get("results") or []
                if result_rows:
                    st.dataframe(pd.DataFrame(result_rows), use_container_width=True)
                return

            from arth_tools.inference.infer import collect_image_items, run_infer

            if source_kind == "Existing run directory":
                source = _require_dir("Run directory", run_dir)
            else:
                source = _require_dir("Bundle directory", bundle)
            if source is None:
                return
            dest = Path(out_csv.strip()) if out_csv.strip() else source / "infer_predictions.csv"
            with st.spinner("Running inference…"):
                summary = run_infer(
                    source,
                    items=collect_image_items(root),
                    out_csv=dest,
                    device="auto",
                )
            st.success(f"Wrote {summary.get('n_images', 0)} predictions → {summary.get('predictions')}")
            pred_path = Path(str(summary.get("predictions") or dest))
            if pred_path.is_file():
                st.dataframe(pd.read_csv(pred_path), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def run_train_panel(tool: ToolEntry) -> None:
    key = tool.tool_id
    st.warning("Training runs in this page and can take a long time. Do not close the tab.")
    dicom_raw = st.text_input("Labelled DICOM folder", key=f"{key}_train_dicom")
    label_csv = st.text_input(
        "Label CSV (optional)",
        key=f"{key}_label_csv",
        help="If set, labels are joined from this spreadsheet (label_source=csv).",
    )
    arches = _architectures()
    default_arch = tool.architecture_id if tool.architecture_id in arches else (arches[0] if arches else "cnn")
    architecture = st.selectbox(
        "Architecture",
        arches or ["cnn"],
        index=(arches or ["cnn"]).index(default_arch) if default_arch in (arches or ["cnn"]) else 0,
        key=f"{key}_arch",
    )
    epochs = st.number_input("Epochs (optional override)", min_value=0, value=0, step=1, key=f"{key}_epochs")

    from arth_tools.data.preprocess import PreprocessRecipe
    from arth_tools.inference.infer import collect_image_items
    from arth_tools.training.config import load_config_yaml

    cfg = load_config_yaml(tool.yaml_path)
    recipe = PreprocessRecipe.from_training_config(cfg)
    state_key = _preview_state_key("train", key)
    seed = int(cfg.seed)

    if st.button("Load preview", key=f"{key}_train_preview"):
        root = _require_dir("Labelled DICOM folder", dicom_raw)
        if root is not None:
            try:
                items = collect_image_items(root)
                rows = sample_preview_rows(items, recipe=recipe, n=PREVIEW_N, seed=seed)
                st.session_state[state_key] = {
                    "root": str(root.resolve()),
                    "paths": [str(r.path) for r in rows],
                    "seed": seed,
                }
                if not rows:
                    st.error("No readable images under that folder.")
            except Exception as exc:
                st.error(str(exc))

    preview_root = Path(dicom_raw.strip()) if dicom_raw.strip() else None
    state = st.session_state.get(state_key)
    ready = preview_root is not None and preview_root.is_dir() and _preview_matches(state, preview_root)
    if ready and state is not None:
        rows = sample_preview_rows(
            state["paths"],
            recipe=recipe,
            n=PREVIEW_N,
            seed=int(state.get("seed", seed)),
            shuffle=False,
        )
        render_preview_grid(rows, show_processed=True, processed_size=recipe.image_size())
    else:
        st.caption("Load a paired preview (decoded vs preprocessed) before training.")

    if st.button("Prepare and train", type="primary", disabled=not ready, key=f"{key}_train_go"):
        root = _require_dir("Labelled DICOM folder", dicom_raw)
        if root is None:
            return
        csv_path = Path(label_csv.strip()) if label_csv.strip() else None
        if csv_path is not None and not csv_path.is_file():
            st.error(f"Label CSV is not a file: {csv_path}")
            return
        try:
            from arth_tools.data.labels import LabelError
            from arth_tools.data.prepare import prepare_from_config
            from arth_tools.training.train import train

            cfg.dicom_root = root
            cfg.architecture_id = architecture
            if csv_path is not None:
                cfg.label_csv = csv_path
                cfg.label_source = "csv"
            if int(epochs) > 0:
                cfg.epochs = int(epochs)
            with st.spinner("Prepare (export + split)…"):
                prepare_from_config(cfg)
            with st.spinner("Training…"):
                result = train(cfg)
            st.success("Training finished.")
            st.json(
                {
                    "run_id": result.get("run_id"),
                    "run_dir": result.get("run_dir"),
                    "bundle_dir": result.get("bundle_dir"),
                    "best_epoch": result.get("best_epoch"),
                    "best_metric": result.get("best_metric"),
                }
            )
        except LabelError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(str(exc))


_UNSET = ""


def render_welcome() -> None:
    st.title("Welcome to arth_tools")
    st.write(
        "arth_tools is a local research library for arthritis imaging. "
        "Use it to grade radiographs, measure hip–knee–ankle alignment, "
        "or train a classifier on labelled DICOMs you provide."
    )
    st.markdown(
        """
Images and labels stay on the machine running this app. The library does not
upload studies. This is research software, not a medical device, and it is
not intended for clinical diagnosis.

Start in the sidebar, one step at a time:

1. **Anatomy** — the joint or region (knee is listed first).
2. **Modality** — X-ray, ultrasound, or MRI. Empty slots stay visible on purpose.
3. **Task** — the clinical question (Kellgren–Lawrence grade, HKA angle, OMERACT synovitis, and so on).
4. **Tool** — a specific implementation. Some tools only infer; native CNN tools can also train.

Infer and Train show a sample of decoded images before anything runs so you can
confirm the files loaded.
        """
    )


def _sidebar_choice(
    label: str,
    ids: list[str],
    names: dict[str, str],
    *,
    key: str,
    placeholder: str,
) -> str:
    options = [_UNSET, *ids]
    return str(
        st.sidebar.selectbox(
            label,
            options,
            format_func=lambda i: placeholder if i == _UNSET else names.get(i, i),
            key=key,
        )
        or _UNSET
    )


def _select_tool() -> ToolEntry | None:
    try:
        catalog = load_catalog()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None
    if not catalog.anatomies:
        st.error("Catalog has no anatomies.")
        return None

    st.sidebar.header("Library")
    st.sidebar.caption("Choose one level at a time.")

    anatomy_ids = [a.anatomy_id for a in catalog.anatomies]
    anatomy_names = {a.anatomy_id: a.anatomy_name for a in catalog.anatomies}
    anatomy_id = _sidebar_choice(
        "Anatomy",
        anatomy_ids,
        anatomy_names,
        key="catalog_anatomy",
        placeholder="Select anatomy…",
    )
    if not anatomy_id:
        return None
    anatomy = next(a for a in catalog.anatomies if a.anatomy_id == anatomy_id)

    modality_ids = [m.modality_id for m in anatomy.modalities]
    modality_names = {m.modality_id: m.modality_name for m in anatomy.modalities}
    modality_id = _sidebar_choice(
        "Modality",
        modality_ids,
        modality_names,
        key=f"catalog_modality_{anatomy_id}",
        placeholder="Select modality…",
    )
    if not modality_id:
        return None
    modality = next(m for m in anatomy.modalities if m.modality_id == modality_id)

    if not modality.tasks:
        st.info(f"No tasks yet for {anatomy.anatomy_name} {modality.modality_name}.")
        return None

    task_ids = [t.task_id for t in modality.tasks]
    task_names = {t.task_id: t.task_name for t in modality.tasks}
    task_id = _sidebar_choice(
        "Task",
        task_ids,
        task_names,
        key=f"catalog_task_{anatomy_id}_{modality_id}",
        placeholder="Select task…",
    )
    if not task_id:
        return None
    task = next(t for t in modality.tasks if t.task_id == task_id)

    if not task.tools:
        st.info(f"No tools yet for {task.task_name}.")
        return None

    tool_ids = [t.tool_id for t in task.tools]
    tool_names = {t.tool_id: t.tool_name for t in task.tools}
    tool_id = _sidebar_choice(
        "Tool",
        tool_ids,
        tool_names,
        key=f"catalog_tool_{anatomy_id}_{modality_id}_{task_id}",
        placeholder="Select tool…",
    )
    if not tool_id:
        return None
    return next(t for t in task.tools if t.tool_id == tool_id)


def main() -> None:
    st.set_page_config(page_title="arth_tools", layout="wide")

    tool = _select_tool()
    if tool is None:
        render_welcome()
        return

    st.title("arth_tools")
    st.header(tool.task_name)
    st.caption(tool.tool_name)

    modes = ["Infer", "Info"]
    if tool.train_enabled:
        modes = ["Infer", "Train", "Info"]
    mode = st.radio("Tool panel", modes, horizontal=True, key=f"{tool.tool_id}_mode")
    if mode == "Info":
        render_info(tool)
    elif mode == "Infer":
        run_infer_panel(tool)
    else:
        run_train_panel(tool)


if __name__ == "__main__":
    main()
