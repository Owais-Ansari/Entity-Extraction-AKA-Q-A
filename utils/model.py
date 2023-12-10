import config
import transformers
import torch.nn as nn

class BERTBaseUncasedTweet(nn.Module):
    def __init__(self,num_classes=2):
        super(BERTBaseUncasedTweet, self).__init__()
        #super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, return_dict = config.return_dict)
        self.lo = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        #not using sentiment
        sequence_output, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # sequence_output : [batch_size, num_tokens,768]
        logits = self.lo(sequence_output) # [batch_size, num_tokens,2]
        
        
        start_logits,end_logits= logits.split(1,dim=-1) # [batch_size, num_tokens,1] , # [batch_size, num_tokens,1]
        start_logits = start_logits.squeeze(2)
        end_logits = end_logits.squeeze(2)
        return start_logits, end_logits
    
    
# '1.10.2+cu111'   
# '0.11.3+cu111' 
# '0.10.2+cu111'

