from __future__ import annotations

from rf_multiview_relation.models.encoder_iq import IQEncoder


class APEncoder(IQEncoder):
    """AP uses the same Conv1D architecture as IQ with independent parameters."""
