<# SOUTHWEST #>

python run_fit.py --config CA_NV_AZ_UT_config *> out/CA_NV_AZ_UT_fit.txt
python run_metrics.py --config CA_NV_AZ_UT_config *> out/CA_NV_AZ_UT_metrics.txt
python run_simulations.py --config CA_NV_AZ_UT_config *> out/CA_NV_AZ_UT_simulations.txt


<# DMV #>

python run_fit.py --config MD_DC_VA_DE_config *> out/MD_DC_VA_DE_fit.txt
python run_metrics.py --config MD_DC_VA_DE_config *> out/MD_DC_VA_DE_metrics.txt
python run_simulations.py --config MD_DC_VA_DE_config *> out/MD_DC_VA_DE_simulations.txt


<# MIDWEST #>

python run_fit.py --config IN_OH_IL_MI_WI_MO_MN_IA_config *> out/IN_OH_IL_MI_WI_MO_MN_IA_fit.txt
python run_metrics.py --config IN_OH_IL_MI_WI_MO_MN_IA_config *> out/IN_OH_IL_MI_WI_MO_MN_IA_metrics.txt
python run_simulations.py --config IN_OH_IL_MI_WI_MO_MN_IA_config *> out/IN_OH_IL_MI_WI_MO_MN_IA_simulations.txt


<# TEXAS #>

python run_fit.py --config TX_config *> out/TX_fit.txt
python run_metrics.py --config TX_config *> out/TX_metrics.txt
python run_simulations.py --config TX_config *> out/TX_simulations.txt


<# SOUTHEAST #>

python run_fit.py --config FL_GA_SC_NC_config *> out/FL_GA_SC_NC_fit.txt
python run_metrics.py --config FL_GA_SC_NC_config *> out/FL_GA_SC_NC_metrics.txt
python run_simulations.py --config FL_GA_SC_NC_config *> out/FL_GA_SC_NC_simulations.txt