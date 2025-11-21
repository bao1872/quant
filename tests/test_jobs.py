import unittest
from datetime import date

import data.jobs as jobs


class TestJobs(unittest.TestCase):
    def test_job_update_bollinger_runs(self):
        today = date.today()
        jobs.job_update_bollinger(today)

    def test_job_update_minute_60m_runs(self):
        today = date.today()
        jobs.job_update_minute_60m(today)

    def test_job_update_minute_15m_runs(self):
        today = date.today()
        jobs.job_update_minute_15m(today)


if __name__ == '__main__':
    unittest.main()
