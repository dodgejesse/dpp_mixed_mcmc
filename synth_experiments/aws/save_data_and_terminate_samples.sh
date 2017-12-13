

CUR_IP=${1}
#SAVE_LOC=/home/ec2-user/projects/dpp_mixed_mcmc/synth_experiments/pickled_data/all_samples
SAVE_LOC=/home/ec2-user/projects/dpp_mixed_mcmc/synth_experiments/pickled_data/dim=1

scp -i /home/ec2-user/projects/dpp_mixed_mcmc/synth_experiments/aws/jesse-key-pair-uswest2.pem -oStrictHostKeyChecking=no $SAVE_LOC/* ec2-user@${CUR_IP}:${SAVE_LOC}


aws ec2 terminate-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)

#SPOT_INST_ID=${2}
#ssh -i /home/ec2-user/projects/dpp_mixed_mcmc/synth_experiments/aws/jesse-key-pair-uswest2.pem -oStrictHostKeyChecking=no ec2-user@${CUR_IP} "aws ec2 terminate-instances --instance-ids ${SPOT_INST_ID}"


