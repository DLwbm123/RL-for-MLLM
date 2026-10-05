from src.experiment import train
import os
if __name__=='__main__': train(os.environ['METHOD'],os.environ.get('RESUME_CHECKPOINT'))
