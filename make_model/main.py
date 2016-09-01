import argparse

import metaprocessing
import vgg16_multi_gpu as vgg

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
                      help='Factor by which learning rate should be decayed after'+\
                      'every decay epochs')
  parser.add_argument('--decay_epochs', type=int, default=30,
                      help='Learning rate to be decayed after every decay epochs')
  parser.add_argument('--no_gpus', type=int, default=1, choices=[1,2],
                      help='Number of GPUs to be used (Data Parallelism). ' + \
                      'no_gpu>2 is not tested')
  parser.add_argument('--checkpoint_dir', default='checkpoints',
                      help='Directory for storing/loading model weights')
  parser.add_argument('--output_file', default='out.txt',
                      help='File to write output')

  args = parser.parse_args()
  
  if args.tra
  #no_classes, no_train, _, _ = metaprocessing.setup_meta()
  #steps_per_epoch = no_train/args.batch_size
  #decay_steps = args.decay_epochs * steps_per_epoch
  
  """
  vgg.run_training(no_classes, args.batch_size, args.epochs, steps_per_epoch
                   args.base_learning_rate, decay_steps,
                   args.decay_factor, args.no_gpus, 'checkpoints')
  """
if __name__ == '__main__':
  main()