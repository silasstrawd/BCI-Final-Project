
modelName = 'speechTransformer_AdamW'

args = {}
args['outputDir'] = '/home/jupyter/neural_seq_decoder-master/logs/' + modelName
args['datasetPath'] = '/home/jupyter/neural_seq_decoder-master/data/data'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.002
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 3 #changed from 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['Warmup Steps'] = 0

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)