import torch


model_file = '../data/experiment/PBCL_TMM/latest.pth'
checkpoint0 = torch.load(model_file, map_location='cuda:0')
checkpoint = \
            {
                # 'TIM': checkpoint0['TIM'],
                'TMM': checkpoint0['TMM']
            }
torch.save(checkpoint, model_file)
