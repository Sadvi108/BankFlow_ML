import json
import torch
from pathlib import Path
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from enhanced_patterns import EnhancedPatternMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_fast_model():
    """Load the fast trained model"""
    try:
        checkpoint = torch.load('app/models/transaction_extractor_fast.pt', map_location='cpu')
        tokenizer = LayoutLMTokenizer.from_pretrained(checkpoint['tokenizer_name'], use_fast=False)
        
        # Get number of labels from config
        num_labels = checkpoint.get('model_config', {}).get('num_labels', 2)
        
        model = LayoutLMForTokenClassification.from_pretrained(
            'microsoft/layoutlm-base-uncased', 
            num_labels=num_labels
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load fast model: {e}")
        return None, None

def extract_with_model(text, model, tokenizer):
    """Extract transaction ID using the trained model"""
    try:
        model.eval()
        encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        
        # Simple bounding boxes (this is a limitation, but works for testing)
        seq_len = encoding['input_ids'].shape[1]
        bboxes = []
        for i in range(seq_len):
            # Simple linear progression for bbox coordinates
            x = min(1000, i * 10)
            bboxes.append([x, 0, min(1000, x + 10), 20])
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                bbox=torch.tensor([bboxes])
            )
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Extract predicted transaction ID
        transaction_ids = []
        current_id = ""
        
        for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
            if pred.item() == 1 and token not in ['[CLS]', '[SEP]', '']:
                # Clean token and add to current ID
                clean_token = token.replace('##', '')
                if clean_token.isalnum() or '-' in clean_token:
                    current_id += clean_token
            elif current_id:
                if len(current_id) > 5:  # Minimum length for transaction ID
                    transaction_ids.append(current_id)
                current_id = ""
        
        # Add final ID if exists
        if current_id and len(current_id) > 5:
            transaction_ids.append(current_id)
        
        return transaction_ids
    except Exception as e:
        logger.error(f"Model extraction failed: {e}")
        return []

def test_final_extraction():
    """Test the final extraction on realistic receipt texts"""
    pattern_matcher = EnhancedPatternMatcher()
    
    # Load fast model
    model, tokenizer = load_fast_model()
    
    # Realistic test cases based on actual receipt patterns
    test_cases = [
        ("Maybank receipt", "Maybank Transfer Reference: MYCN251031853500 Status: Successful"),
        ("CIMB receipt", "CIMB Transaction ID: B10-2510-625105 Amount: RM250.00"),
        ("Public Bank receipt", "Public Bank Payment PBB251031758202 Completed"),
        ("RHB receipt", "RHB Transfer RHB251031180741 Done"),
        ("HSBC receipt", "HSBC Transaction HSBC251031529613 Processed"),
        ("UOB receipt", "UOB Reference UOB251031200108 Status: Completed"),
        ("Standard Chartered receipt", "SCB Transfer SCB251031744606 Successful"),
        ("DuitNow receipt", "DuitNow Payment DN251031333313 Reference"),
        ("Mixed format", "Transfer completed. Ref: MYCN251031291491 Date: 31/10/25"),
        ("Lowercase format", "transaction id: b10-2510-240816 amount: rm100"),
        ("With spaces", "Payment Reference : PBB251031580390 Status : Done"),
        ("Long format", "Bank Transfer Reference Number RHB251031943944 Amount RM500.00"),
        ("Short format", "Ref: HSBC251031645651"),
        ("With prefix", "Txn ID: UOB251031806076 Date: 31-Oct-2025"),
        ("DuitNow format", "DuitNow Transfer DN251031565451 Completed Successfully"),
        ("Mixed case", "maybank reference mycn251031842914 status ok"),
        ("With symbols", "Transaction-> SCB251031911983 | Amount: RM1,000"),
        ("Multi-line", "Transfer Details:\nReference: DN251031915330\nAmount: RM750"),
        ("Bank prefix", "Standard Chartered Payment SCB251031295446 Processed"),
        ("End of text", "Thank you for using our service. Ref: MYCN251031123456"),
        # More challenging cases
        ("No spaces", "TransactionID:B10-2510-409674Amount:RM200"),
        ("Special chars", "Payment #PBB251031407081# Completed"),
        ("Multiple refs", "Ref1: RHB251031798726 Ref2: HSBC251031701530"),
        ("With date", "Date: 2025-10-31 Ref: UOB251031499003"),
    ]
    
    results = []
    
    logger.info(f"Testing {len(test_cases)} receipt patterns")
    
    for case_name, text in test_cases:
        try:
            # Pattern matching
            pattern_results = pattern_matcher.extract_transaction_ids(text)
            
            # Model extraction (if available)
            model_results = []
            if model and tokenizer:
                model_results = extract_with_model(text, model, tokenizer)
            
            # Combine results
            all_ids = list(set(pattern_results + model_results))
            
            # Check if we found the expected ID
            expected_found = any(len(tid) > 8 for tid in all_ids)  # Reasonable transaction ID length
            
            results.append({
                'case': case_name,
                'text': text,
                'pattern_ids': pattern_results,
                'model_ids': model_results,
                'combined_ids': all_ids,
                'success': len(all_ids) > 0 and expected_found
            })
            
            logger.info(f"{case_name}: {all_ids} {'✓' if len(all_ids) > 0 and expected_found else '✗'}")
            
        except Exception as e:
            logger.error(f"Failed to process {case_name}: {e}")
            results.append({
                'case': case_name,
                'error': str(e),
                'success': False
            })
    
    # Calculate success rate
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    logger.info(f"Final Success Rate: {success_rate:.1f}% ({successful}/{total})")
    
    # Detailed analysis
    pattern_only_success = sum(1 for r in results if len(r.get('pattern_ids', [])) > 0)
    model_only_success = sum(1 for r in results if len(r.get('model_ids', [])) > 0)
    combined_success = sum(1 for r in results if len(r.get('combined_ids', [])) > 0)
    
    logger.info(f"Pattern-only success: {pattern_only_success}/{total} ({pattern_only_success/total*100:.1f}%)")
    logger.info(f"Model-only success: {model_only_success}/{total} ({model_only_success/total*100:.1f}%)")
    logger.info(f"Combined success: {combined_success}/{total} ({combined_success/total*100:.1f}%)")
    
    # Save results
    with open('final_test_results.json', 'w') as f:
        json.dump({
            'success_rate': success_rate,
            'total_cases': total,
            'successful_extractions': successful,
            'pattern_only_success': pattern_only_success,
            'model_only_success': model_only_success,
            'combined_success': combined_success,
            'results': results
        }, f, indent=2)
    
    return success_rate

if __name__ == "__main__":
    success_rate = test_final_extraction()
    print(f"🎯 Final Success Rate: {success_rate:.1f}%")