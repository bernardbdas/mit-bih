"""
Data processing and download APIs for the MIT-BIH dataset.
"""

from mit_bih.data.download import DATABASES, download_database
from mit_bih.data.preprocess import (
    AAMI_MAPPING,
    bandpass_filter,
    segment_record,
    process_and_segment_records,
    split_by_task,
    make_continual_loader,
    create_federated_continual_clients
)

__all__ = [
    "DATABASES",
    "download_database",
    "AAMI_MAPPING",
    "bandpass_filter",
    "segment_record",
    "process_and_segment_records",
    "split_by_task",
    "make_continual_loader",
    "create_federated_continual_clients"
]
