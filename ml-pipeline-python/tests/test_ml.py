from vidgrep_ml.concepts import extract_concepts, vector_for_labels, vector_for_text
from vidgrep_ml.demo import demo_video
from vidgrep_ml.models import Frame
from vidgrep_ml.service import IndexService
from vidgrep_ml.vector_store import InMemoryStore, cosine


def test_shared_space_alignment():
    # a text query and an image with the same concepts should be ~identical
    img = vector_for_labels(["person", "red jacket"])
    txt = vector_for_text("a person in a red jacket")
    assert cosine(img, txt) > 0.99
    # an unrelated frame should score much lower
    other = vector_for_labels(["truck", "night"])
    assert cosine(other, txt) < 0.5


def test_extract_concepts():
    assert "red jacket" in extract_concepts("find the person in a red jacket")
    assert "turning left" in extract_concepts("cars turning left please")
    assert extract_concepts("完全 unrelated gibberish") == []


def test_vector_store_ranks_by_cosine():
    store = InMemoryStore()
    store.upsert("v", 0, vector_for_labels(["car"]), ["car"])
    store.upsert("v", 1000, vector_for_labels(["person", "red jacket"]), ["person", "red jacket"])
    assert store.count("v") == 2
    hits = store.search("v", vector_for_text("red jacket"), top_k=2)
    assert hits[0][0].timestamp_ms == 1000
    assert hits[0][1] >= hits[1][1]


def test_end_to_end_search_finds_right_frame():
    svc = IndexService()
    svc.index_frames(demo_video("demo"))

    top = svc.search("demo", "person in a red jacket", top_k=3)
    assert top[0].timestamp_ms == 3000  # the frame labeled exactly [person, red jacket]
    assert top[0].box is not None and len(top[0].box) == 4

    turning = svc.search("demo", "a car turning left", top_k=3)
    assert turning[0].timestamp_ms == 1000

    bags = svc.search("demo", "someone carrying a backpack", top_k=3)
    assert any("backpack" in r.labels for r in bags)


def test_search_returns_sorted_topk():
    svc = IndexService()
    svc.index_frames(demo_video("demo"))
    res = svc.search("demo", "night", top_k=4)
    assert len(res) == 4
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)


def test_frame_without_labels_uses_tensor():
    import base64

    svc = IndexService()
    tensor = base64.b64encode(bytes([120] * 4096)).decode()
    frame = Frame("v2", 0, tensor_b64=tensor, labels=[])
    vec = svc.index_frame(frame)
    assert len(vec) == 512
    assert any(abs(x) > 0 for x in vec)
