from src.experiment import audit
import os
if __name__=='__main__': audit(os.environ.get('TAG','B0'))
