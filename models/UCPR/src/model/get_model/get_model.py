
from models.UCPR.src.parser import parse_args


from models.UCPR.src.model.lstm_base.model_lstm_mf_emb import AC_lstm_mf_dummy
from models.UCPR.src.model.baseline.baseline import ActorCritic

from models.UCPR.src.model.UCPR import UCPR

from models.UCPR.src.env.env import *
from models.UCPR.utils import *


args = parse_args()

# ********************* model select *****************************
if args.model == 'lstm': 
    Memory_Model = AC_lstm_mf_dummy
elif args.model == 'UCPR':
    Memory_Model = UCPR

elif args.model == 'baseline':
    Memory_Model = ActorCritic
print(args.model)
# ********************* model select *****************************

KGEnvironment = BatchKGEnvironment#BatchKGEnvironment if args.dataset == ML1M else BatchKGEnvironment 

# ********************* BatchKGEnvironment ************************