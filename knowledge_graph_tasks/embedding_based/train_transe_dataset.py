import sys
sys.path.append('../knowledge_graph_tasks/embedding_based/openke')
# import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.model import TransH
from openke.module.model import TransD
from openke.module.model import DistMult
from openke.module.model import RESCAL
from openke.module.model import ComplEx
from openke.module.model import Analogy
from openke.module.model import SimplE
from openke.module.loss import MarginLoss
from openke.module.loss import SigmoidLoss
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader



def load_data_for_training(args):
	# dataloader for training
	datasetname = args.datasetname
	
	model_parameters = {
		"transd": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		},
		"transh": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		},
		"distmult": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		},
		"transe": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		},
		"rescal": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		},
		"complEx": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		},
		"analogy": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		},
		"simple": {
			"in_path": f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
			"nbatches": 100,
			"threads": 8,
			"sampling_mode": "normal",
			"bern_flag": 1,
			"filter_flag": 1,
			"neg_ent": 25,
			"neg_rel": 0
		}
	}
	params = model_parameters[args.LP_model]


	train_dataloader = TrainDataLoader(
		in_path = params["in_path"],
		nbatches = params.get("nbatches", 100),
		threads = params["threads"],
		sampling_mode = params["sampling_mode"],
		bern_flag = params["bern_flag"],
		filter_flag = params["filter_flag"],
		neg_ent = params["neg_ent"],
		neg_rel = params["neg_rel"]
	)
	# train_dataloader = TrainDataLoader(
	# 	in_path = f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/",
	# 	nbatches = 100,
	# 	threads = 8,
	# 	sampling_mode = "normal",
	# 	bern_flag = 1,
	# 	filter_flag = 1,
	# 	neg_ent = 25,
	# 	neg_rel = 0)

	# dataloader for test
	test_dataloader = TestDataLoader(f"../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/", "link")
	return train_dataloader, test_dataloader


def init_embedding_based_model(args,train_dataloader):
	# define the model
	model = args.LP_model
	if(model == 'simple'):
		# define the model
		simple = SimplE(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim = 200
		)

		# define the loss function
		ns_model = NegativeSampling(
			model = simple, 
			loss = SoftplusLoss(),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = 1.0
		)
		return simple,ns_model
	elif(model == 'analogy'):
		# define the model
		analogy = Analogy(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim = 200
		)

		# define the loss function
		ns_model = NegativeSampling(
			model = analogy, 
			loss = SoftplusLoss(),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = 1.0
		)
		return analogy,ns_model
	elif(model == 'complEx'):
		# define the model
		complEx = ComplEx(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim = 200
		)

		# define the loss function
		ns_model = NegativeSampling(
			model = complEx, 
			loss = SoftplusLoss(),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = 1.0
		)
		return complEx,ns_model
	elif (model == 'rescal'):
		rescal = RESCAL(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim = 50
		)

		ns_model = NegativeSampling(
			model = rescal, 
			loss = MarginLoss(margin = 1.0),
			batch_size = train_dataloader.get_batch_size(), 
		)
		return rescal,ns_model
	elif(model == 'transe'):
		transe = TransE(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim = 200,
			p_norm = 1,
			norm_flag = True)


		# define the loss function
		ns_model = NegativeSampling(
			model = transe,
			loss = MarginLoss(margin = 5.0),
			batch_size = train_dataloader.get_batch_size()
		)
		return transe, ns_model
	elif(model == 'transh'):
		transh = TransH(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim = 200, 
			p_norm = 1, 
			norm_flag = True)

		# define the loss function
		ns_model = NegativeSampling(
			model = transh, 
			loss = MarginLoss(margin = 4.0),
			batch_size = train_dataloader.get_batch_size()
		)
		return transh,ns_model
	elif (model == 'distmult'):
		distmult = DistMult(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim = 200
		)

		# define the loss function
		ns_model = NegativeSampling(
			model = distmult, 
			loss = SoftplusLoss(),
			batch_size = train_dataloader.get_batch_size(), 
			regul_rate = 1.0
		)
		return distmult,ns_model
	elif (model == 'transd'):
		transd = TransD(
			ent_tot = train_dataloader.get_ent_tot(),
			rel_tot = train_dataloader.get_rel_tot(),
			dim_e = 200, 
			dim_r = 200, 
			p_norm = 1, 
			norm_flag = True)


		# define the loss function
		ns_model = NegativeSampling(
			model = transd, 
			loss = MarginLoss(margin = 4.0),
			batch_size = train_dataloader.get_batch_size()
		)
		return transd,ns_model

def train_embedding_based_models(model, ns_model, train_dataloader, test_dataloader,args):
	datasetname = args.datasetname
	train_times = args.train_times
	LP_model = args.LP_model
	if(LP_model == 'simple'):
		# train the model
		print("train_embedding_based_models: ",LP_model)
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1
	elif(LP_model == 'analogy'):
		# train the model
		print("train_embedding_based_models: ",LP_model)
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1
	elif(LP_model == 'complEx'):
		# train the model
		print("train_embedding_based_models: ",LP_model)
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1
	elif(LP_model == 'rescal'):
		# train the model
		print("train_embedding_based_models: ",LP_model)
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 0.1, use_gpu = True, opt_method = "adagrad")
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1
	elif(LP_model == 'transe'):
		# train the model
		print("train_embedding_based_models: ",LP_model)
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 1.0, use_gpu = True)
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1
	elif(LP_model == 'transh'):
		# train the model
		print("train_embedding_based_models: ",LP_model)
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 0.5, use_gpu = True)
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1
	elif(LP_model == 'transd'):
		# train the model
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 1.0, use_gpu = True)
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1
	elif(LP_model == 'distmult'):
		# train the model
		print("train_embedding_based_models: ",LP_model)
		trainer = Trainer(model = ns_model, data_loader = train_dataloader, train_times = train_times, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
		trainer.run()
		model.save_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')

		# test the model
		model.load_checkpoint(f'../knowledge_graph_tasks/embedding_based/checkpoint/{LP_model}_{datasetname}.ckpt')
		tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)
		return mrr, mr, hit10, hit3, hit1