from src.experiment import train
import os
if __name__=='__main__': train(os.environ.get('METHOD','B1'),os.environ.get('RESUME_CHECKPOINT'))
