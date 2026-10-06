"""CPU-only check of the actual process-group deadline/failure watchdog."""
import subprocess
import sys
import time
import unittest
from scripts.launch_rsna_v3_g1 import watch


class WatchdogTest(unittest.TestCase):
    def test_deadline_and_sibling_failure_stop_owned_workers(self):
        for failing in (False, True):
            workers = []
            for entry in (['import time; time.sleep(30)', 'raise SystemExit(7)'] if failing
                          else ['import time; time.sleep(30)']):
                process = subprocess.Popen([sys.executable, '-'], stdin=subprocess.PIPE,
                                           start_new_session=True)
                process.stdin.write(entry.encode()); process.stdin.close()
                workers.append((process, {'started': time.time(), 'finished': None}))
            started = time.time()
            result = watch(workers, started + (10 if failing else .2))
            self.assertEqual(result, 'failed_worker' if failing else 'partial_budget')
            self.assertLess(time.time() - started, 5)
            self.assertTrue(all(p.poll() is not None and r['finished'] >= r['started'] for p, r in workers))


if __name__ == '__main__':
    unittest.main()
