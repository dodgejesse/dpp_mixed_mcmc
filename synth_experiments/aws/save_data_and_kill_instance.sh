

CUR_IP=${1}
SPOT_INST_ID=${2}
SAVE_LOC=/home/ec2-user/projects/dpp_mixed_mcmc/synth_experiments/pickled_data/origin_center_data

scp -i /home/ec2-user/projects/dpp_mixed_mcmc/synth_experiments/aws/jesse-key-pair-uswest2.pem -oStrictHostKeyChecking=no $SAVE_LOC/* ec2-user@${CUR_IP}:${SAVE_LOC}
ssh -i /home/ec2-user/projects/dpp_mixed_mcmc/synth_experiments/aws/jesse-key-pair-uswest2.pem -oStrictHostKeyChecking=no ec2-user@${CUR_IP} "aws ec2 terminate-instances --instance-ids ${SPOT_INST_ID}"


