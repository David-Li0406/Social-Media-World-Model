from smwm.baselines import registry  # noqa: F401  (triggers registration)


def test_constant_baselines_registered():
    names = set(registry.list_baselines())
    assert {"constant_mean", "constant_median", "feature_gbdt",
            "retrieval_tfidf", "retrieval_sbert", "encoder", "llm"} <= names


def test_constant_mean_predicts_dict():
    cls = registry.get("constant_mean")
    bl = cls()
    bl.fit([
        {"ground_truth": {"score": 10, "width": 1, "controversiality": 0, "reply_summary": ""}},
        {"ground_truth": {"score": 30, "width": 3, "controversiality": 1, "reply_summary": ""}},
    ])
    pred = bl.predict({})
    assert set(pred) >= {"score", "width", "controversiality", "reply_summary"}
    assert pred["score"] == 20
    assert pred["width"] == 2
