
from UCPR.src.parser import parse_args


from UCPR.src.model.lstm_base.model_lstm_mf_emb import AC_lstm_mf_dummy
from UCPR.src.model.baseline.baseline import ActorCritic

from UCPR.src.model.UCPR import UCPR

from UCPR.src.env.env import *
from UCPR.utils import *


args = parse_args()

# ********************* model select *****************************
if args.model == 'lstm': 
    Memory_Model = AC_lstm_mf_dummy
elif args.model == 'UCPR':
    Memory_Model = UCPR

elif args.model == 'baseline':
    Memory_Model = ActorCritic

# ********************* model select *****************************

KGEnvironment = BatchKGEnvironment#BatchKGEnvironment if args.dataset == ML1M else BatchKGEnvironment 

# ********************* BatchKGEnvironment ************************