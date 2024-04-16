from bids import BIDSLayout
from bids.reports import BIDSReport
from bids.tests import get_test_data_path


def get_subject_list(bids_dir):
    layout = BIDSLayout(bids_dir, derivatives=True)
    subjects = layout.get_subjects()
    return subjects

def get_report(bids_dir, derivatives=False):
    report = BIDSReport(BIDSLayout(bids_dir))
    counter = report.generate()
    main_report = counter.most_common()[0][0]
    return main_report