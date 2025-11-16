import unittest
from datetime import date

import data.jobs as jobs


class TestJobs(unittest.TestCase):
    def test_job_update_bollinger_runs(self):
        today = date.today()
        jobs.job_update_bollinger(today)


if __name__ == '__main__':
    unittest.main()
