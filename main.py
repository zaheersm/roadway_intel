import argparse
import sys

import settings
import roadway as rd

# Functions for redirecting and restoring stdout
def _redirect_stdout(path):
  orig_stdout = sys.stdout
  f = open(path, 'w', 0)
  sys.stdout = f
  return orig_stdout, f

def _restore_stdout(orig_stdout, f):
  sys.stdout = orig_stdout
  f.close()

def main():

  parser = argparse.ArgumentParser('Fine tunes a VGG16 classification ' + \
                                    'model on CompCars dataset')
  group = parser.add_mutually_exclusive_group()
  group.add_argument("-t", "--training", action="store_true", help="Training")
  group.add_argument("-e", "--evaluation", action="store_true", help="Evaluation")
  parser.add_argument('--batch_size', type=int, choices=[30,40,50,60,70],
                      default=30, help='Batch Size for training/eval. '+ \
                      'Rule of thumb: Batch of size ~30 for each GPU')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs for running the training routine')
  parser.add_argument('--base_learning_rate', type=float, default=0.0001,
                      choices=[0.01,0.001,0.0001, 0.00001],
                      help='Learning Rate to be used for fine-tuning')
  parser.add_argument('--decay_factor', type=float, 
                      default=0.1, choices=[0.1, 0.01], 
                      help='Learning rate to be decayed by decay_factor')
  parser.add_argument('--decay_epochs', type=int, default=30,
                      help='Learning rate to be decayed after every decay epochs')
  parser.add_argument('--no_gpus', type=int, default=1, choices=[1,2],
                      help='Number of GPUs to be used (Data Parallelism). ' + \
                      'no_gpu>2 is not tested')
  parser.add_argument('--setup_meta', action='store_true',
                      help='Generate meta files for train/valid/test split')

  args = parser.parse_args()

  # Tensorflow blurts out tons of log messages
  # Therefore, we will redirect stdout to file specified by
  # settings.TRAINING_OUTPUT/settings.EVALUATION_OUTPUT
  output_file = settings.TRAINING_OUTPUT if args.training\
                                         else settings.EVALUATION_OUTPUT
  orig_stdout, f = _redirect_stdout(output_file)
  try:
    if args.setup_meta:
      # If you do this step, make sure your previous train/valid/test split
      # metadatas are backed-up as this would overwrite it
      no_classes, no_train, _, _ = rd.metaprocessing.setup_meta()
    else:
      no_classes = rd.metaprocessing.get_no_classes()
      no_train = rd.metaprocessing.get_no_training_samples()

    if args.training == True:
      print ('Training')
      steps_per_epoch = no_train/args.batch_size
      decay_steps = args.decay_epochs * steps_per_epoch

      print ('Configuration:\nNO_CLASSES: %d\nBATCH_SIZE: %d\nEPOCHS: %d\n'\
             'STEPS_PER_EPOCH: %d\nBASE_LEARNING_RATE: %f\nDECAY_STEPS: %d\n'\
             'DECAY_FACTOR: %.2f\nNO_GPUS: %d\nCHECKPOINT_DIR: %s\n' %
             (no_classes, args.batch_size, args.epochs, steps_per_epoch,
              args.base_learning_rate, decay_steps, args.decay_factor,
              args.no_gpus, settings.CHECKPOINT_DIR))

      rd.vgg16.train.run_training(no_classes, args.batch_size, args.epochs,
                                  steps_per_epoch, args.base_learning_rate,
                                  decay_steps, args.decay_factor, args.no_gpus,
                                  settings.CHECKPOINT_DIR)
    else:
      print ('Evaluating')
      rd.vgg16.evaluate.run_evaluation(no_classes,
                                       args.batch_size,
                                       settings.CHECKPOINT_DIR)
  finally:
    _restore_stdout(orig_stdout, f)

if __name__ == '__main__':
  main()
