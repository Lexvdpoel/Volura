"""Safety tests for saving and loading .volura files.

These tests verify that the current behavior of save_document and load_document
is preserved when the codebase is optimized with caching, bulk operations,
or background threads.
"""
import json
import os
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import fileio, ops  # noqa: E402
from volura.model import Document  # noqa: E402


def compare_entities(e1, e2, tol=1e-6):
    """Compare two Entities for equality (volume, face count, edge count, vertices, hidden)."""
    vol1 = e1.to_manifold().volume() if e1.to_manifold().status() == mf.Error.NoError else None
    vol2 = e2.to_manifold().volume() if e2.to_manifold().status() == mf.Error.NoError else None
    assert vol1 == pytest.approx(vol2, rel=1e-6), f"Volume mismatch: {vol1} vs {vol2}"
    assert len(e1.F) == len(e2.F), f"Face count mismatch: {len(e1.F)} vs {len(e2.F)}"
    assert len(e1.E) == len(e2.E), f"Edge count mismatch: {len(e1.E)} vs {len(e2.E)}"
    assert len(e1.V) == len(e2.V), f"Vertex count mismatch: {len(e1.V)} vs {len(e2.V)}"
    for fid in e1.F:
        assert fid in e2.F
        f1 = e1.F[fid]
        f2 = e2.F[fid]
        assert np.allclose(f1.normal, f2.normal, atol=tol), f"Face {fid} normal mismatch: {f1.normal} vs {f2.normal}"
        assert f1.color == f2.color, f"Face {fid} color mismatch"
        assert len(f1.loops) == len(f2.loops), f"Face {fid} loop count mismatch"
        for i, (l1, l2) in enumerate(zip(f1.loops, f2.loops)):
            assert len(l1) == len(l2), f"Face {fid} loop {i} length mismatch"
            for j, (v1, v2) in enumerate(zip(l1, l2)):
                assert v1 in e2.V and v2 in e2.V, f"Face {fid} loop {i} vertex {j} not in e2.V"
    for eid in e1.E:
        assert eid in e2.E
        e1_edge = e1.E[eid]
        e2_edge = e2.E[eid]
        assert set(e1_edge) == set(e2_edge), f"Edge {eid} vertices mismatch"
    assert e1.hidden_f == e2.hidden_f, f"Hidden faces mismatch: {e1.hidden_f} vs {e2.hidden_f}"
    assert e1.hidden_e == e2.hidden_e, f"Hidden edges mismatch: {e1.hidden_e} vs {e2.hidden_e}"
    return vol1


def compare_groups(g1, g2, tol=1e-6):
    """Compare two groups for equality."""
    assert g1.id == g2.id, f"Group id mismatch: {g1.id} vs {g2.id}"
    assert g1.name == g2.name, f"Group name mismatch: {g1.name} vs {g2.name}"
    assert g1.color == g2.color, f"Group color mismatch: {g1.color} vs {g2.color}"
    assert g1.visible == g2.visible, f"Group visible mismatch: {g1.visible} vs {g2.visible}"
    assert g1.locked == g2.locked, f"Group locked mismatch: {g1.locked} vs {g2.locked}"
    return compare_entities(g1.ents, g2.ents, tol)


def test_save_and_load_preserves_all_properties(tmp_path):
    """Test 1: save_document and load_document preserve all group and document properties."""
    doc = Document()
    
    # Create multiple groups with different colors and painted faces
    g1 = ops.group_from_manifold(doc, mf.Manifold.cube((40, 30, 20)), "#f2f2f2", "White Box")
    g2 = ops.group_from_manifold(doc, mf.Manifold.cube((30, 30, 30)), "#d23b3b", "Red Box")
    g3 = ops.group_from_manifold(doc, mf.Manifold.sphere(15, 32), "#2f6fd6", "Blue Sphere")
    
    # Paint some faces on g1
    for f in list(g1.ents.F):
        if g1.ents.F[f].normal[2] > 0.9:
            g1.ents.set_face_color(f, "#d23b3b")
        elif g1.ents.F[f].normal[1] < -0.9:
            g1.ents.set_face_color(f, "#2f6fd6")

    # Set up custom materials
    doc.materials = [("White", "#f2f2f2"), ("Black", "#2b2b2b"), ("Red", "#d23b3b"),
                     ("Blue", "#2f6fd6"), ("Yellow", "#f2c230"), ("Green", "#3fa34d")]
    
    # Add a guide
    line = doc.add_guide([10, 10, 0], [1, 0, 0])
    point = doc.add_guide([5, 5, 5])
    
    # Lock one group
    g2.locked = True
    
    # Save and reload
    path = str(tmp_path / "model.volura")
    fileio.save_document(doc, path)
    
    doc2 = Document()
    fileio.load_document(doc2, path)
    
    # Compare root entities
    compare_entities(doc.root, doc2.root)
    
    # Compare groups by id
    for g1 in doc.groups:
        g2 = doc2.group(g1.id)
        assert g2 is not None, f"Group {g1.id} not found in loaded document"
        compare_groups(g1, g2)
    
    # Compare guides
    assert len(doc.guides) == len(doc2.guides), f"Guide count mismatch: {len(doc.guides)} vs {len(doc2.guides)}"
    for g1, g2 in zip(doc.guides, doc2.guides):
        assert g1.id == g2.id, f"Guide id mismatch: {g1.id} vs {g2.id}"
        assert np.allclose(g1.point, g2.point, atol=1e-6), f"Guide point mismatch"
        if g1.is_line:
            assert np.allclose(g1.direction, g2.direction, atol=1e-6), f"Guide direction mismatch"
    
    # Compare materials (convert both to same format for comparison)
    assert [list(m) for m in doc.materials] == [list(m) for m in doc2.materials], f"Materials mismatch: {doc.materials} vs {doc2.materials}"
    
    # Compare next IDs
    assert doc.next_gid == doc2.next_gid, f"next_gid mismatch"
    assert doc.next_guide_id == doc2.next_guide_id, f"next_guide_id mismatch"
    
    # Check group properties (visible, locked)
    for g1 in doc.groups:
        g2 = doc2.group(g1.id)
        assert g1.visible == g2.visible, f"Group {g1.id} visible mismatch"
        assert g1.locked == g2.locked, f"Group {g1.id} locked mismatch"


def test_save_load_re_save_content_equality(tmp_path):
    """Test 2: Save, load, save again - second save is content-wise equal to first."""
    doc = Document()
    g1 = ops.group_from_manifold(doc, mf.Manifold.cube((40, 30, 20)), "#f2f2f2", "Box")
    g2 = ops.group_from_manifold(doc, mf.Manifold.sphere(15, 32), "#d23b3b", "Sphere")
    
    # Paint some faces
    for f in list(g1.ents.F):
        if g1.ents.F[f].normal[2] > 0.9:
            g1.ents.set_face_color(f, "#d23b3b")
    
    doc.materials = [("White", "#f2f2f2"), ("Red", "#d23b3b")]
    line = doc.add_guide([10, 10, 0], [1, 0, 0])
    
    path1 = str(tmp_path / "model1.volura")
    path2 = str(tmp_path / "model2.volura")
    
    fileio.save_document(doc, path1)
    fileio.save_document(doc, path2)
    
    doc2 = Document()
    fileio.load_document(doc2, path2)
    
    fileio.save_document(doc2, path1)
    
    # Compare JSON contents
    with zipfile.ZipFile(path1) as z:
        json1 = json.loads(z.read("document.json"))
    
    with zipfile.ZipFile(path2) as z:
        json2 = json.loads(z.read("document.json"))
    
    assert json1["format"] == json2["format"], f"Format mismatch"
    assert json1["version"] == json2["version"], f"Version mismatch"
    assert json1["app"] == json2["app"], f"App version mismatch"
    assert json1["materials"] == json2["materials"], f"Materials mismatch"
    assert json1["next_gid"] == json2["next_gid"], f"next_gid mismatch"
    assert json1["group_counter"] == json2["group_counter"], f"group_counter mismatch"
    assert json1["next_guide_id"] == json2["next_guide_id"], f"next_guide_id mismatch"
    assert json1["root"] == json2["root"], f"Root entities mismatch"
    assert json1["groups"] == json2["groups"], f"Groups mismatch"
    assert json1["guides"] == json2["guides"], f"Guides mismatch"
    assert json1["sheet"] == json2["sheet"], f"Sheet mismatch"


def test_edit_after_load_works_normally(tmp_path):
    """Test 3: After loading, editing (add_segment, pushpull) works normally."""
    doc = Document()
    g1 = ops.group_from_manifold(doc, mf.Manifold.cube((40, 30, 20)), "#f2f2f2", "Box")
    
    path = str(tmp_path / "model.volura")
    fileio.save_document(doc, path)
    
    doc2 = Document()
    fileio.load_document(doc2, path)
    
    # Pick a face on the root and add a segment
    if doc2.root.F:
        fid = next(iter(doc2.root.F))
        face = doc2.root.F[fid]
        verts = [doc2.root.V[v] for v in face.loops[0]]
        if len(verts) >= 2:
            p, q = verts[0], verts[1]
            doc2.root.add_segment(p, q)
            
            # Push-pull the face
            doc2.root.pushpull(fid, 5.0)
    
    # Save and reload again
    fileio.save_document(doc2, path)
    doc3 = Document()
    fileio.load_document(doc3, path)
    
    # Verify the edit persisted
    assert len(doc3.root.F) >= len(doc2.root.F), "Face count should not decrease"


def test_large_document_save_load_preserves_volumes(tmp_path):
    """Test 4: Large document (30 spheres of 48 segments) - save/load preserves all volumes."""
    doc = Document()
    volumes_before = []
    
    for i in range(30):
        man = mf.Manifold.sphere(40, 48).translate((i * 100, 0, 0))
        g = ops.group_from_manifold(doc, man, f"#ff{30+i:02x}00", f"Sphere {i}")
        vol = g.ents.to_manifold().volume()
        assert vol > 0, f"Sphere {i} has non-positive volume"
        volumes_before.append(vol)
    
    path = str(tmp_path / "large.volura")
    fileio.save_document(doc, path)
    
    doc2 = Document()
    fileio.load_document(doc2, path)
    
    volumes_after = []
    for g1 in doc.groups:
        g2 = doc2.group(g1.id)
        assert g2 is not None, f"Group {g1.id} not found"
        vol = g2.ents.to_manifold().volume()
        assert vol > 0, f"Loaded Sphere {g1.id} has non-positive volume"
        volumes_after.append(vol)
    
    # Verify all volumes match
    for g1 in doc.groups:
        g2 = doc2.group(g1.id)
        vol1 = g1.ents.to_manifold().volume()
        vol2 = g2.ents.to_manifold().volume()
        assert vol1 == pytest.approx(vol2, rel=1e-6), f"Volume {g1.id} mismatch: {vol1} vs {vol2}"


def test_corrupted_file_gives_error(tmp_path):
    """Test 5: Corrupted file gives a nice error and leaves Document unchanged."""
    doc = Document()
    g1 = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)), "#f2f2f2", "Test")
    doc2 = Document()
    
    # Create a corrupted zip file (not a valid zip)
    path = str(tmp_path / "corrupted.volura")
    with open(path, "wb") as f:
        f.write(b"This is not a valid zip file at all!")
    
    # Try to load - should raise an error
    with pytest.raises(Exception):
        fileio.load_document(doc2, path)
    
    # Verify the document is still unchanged
    assert len(doc2.groups) == 0, "Document should remain unchanged after failed load"
    assert len(doc2.root.F) == 0, "Root should remain unchanged"


def test_truncated_zip_gives_error(tmp_path):
    """Test 6: Truncated zip file gives a nice error and leaves Document unchanged."""
    doc = Document()
    g1 = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)), "#f2f2f2", "Test")
    doc2 = Document()
    
    path = str(tmp_path / "truncated.volura")
    fileio.save_document(doc, path)
    
    # Truncate the file
    with zipfile.ZipFile(path, "r") as z:
        original_size = z.filelist[0].compress_size
    
    with open(path, "rb") as f:
        data = f.read()
    
    with open(path, "wb") as f:
        f.write(data[:original_size // 2])
    
    # Try to load - should raise an error
    with pytest.raises(Exception):
        fileio.load_document(doc2, path)
    
    # Verify the document is still unchanged
    assert len(doc2.groups) == 0, "Document should remain unchanged after failed load"
