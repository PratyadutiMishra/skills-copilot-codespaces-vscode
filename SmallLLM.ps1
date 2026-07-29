# Small Language Model Implementation in PowerShell
# Fully corrected - simplified version without constructor issues

class RandomHelper {
    static [System.Random]$rng = [System.Random]::new(42)
    
    static [double] NextGaussian() {
        $u1 = [RandomHelper]::rng.NextDouble()
        $u2 = [RandomHelper]::rng.NextDouble()
        return [Math]::Sqrt(-2 * [Math]::Log($u1)) * [Math]::Cos(2 * [Math]::PI * $u2)
    }
}

class MatrixOps {
    static [double[][]] CreateMatrix([int]$rows, [int]$cols, [double]$scale) {
        [double[][]]$matrix = New-Object 'double[][]' $rows
        
        for ($i = 0; $i -lt $rows; $i++) {
            $matrix[$i] = New-Object 'double[]' $cols
            for ($j = 0; $j -lt $cols; $j++) {
                $matrix[$i][$j] = [RandomHelper]::NextGaussian() * $scale
            }
        }
        return $matrix
    }

    static [double[][]] Multiply([double[][]]$A, [double[][]]$B, [double]$scale) {
        $rows = $A.Count
        $cols = $B[0].Count
        $inner = $B.Count
        [double[][]]$result = New-Object 'double[][]' $rows
        
        for ($i = 0; $i -lt $rows; $i++) {
            $result[$i] = New-Object 'double[]' $cols
            for ($j = 0; $j -lt $cols; $j++) {
                $sum = 0
                for ($k = 0; $k -lt $inner; $k++) {
                    $sum += $A[$i][$k] * $B[$k][$j]
                }
                $result[$i][$j] = $sum * $scale
            }
        }
        return $result
    }

    static [double[][]] Softmax([double[][]]$matrix) {
        $rows = $matrix.Count
        $cols = $matrix[0].Count
        [double[][]]$result = New-Object 'double[][]' $rows
        
        for ($i = 0; $i -lt $rows; $i++) {
            $result[$i] = New-Object 'double[]' $cols
            $maxVal = ($matrix[$i] | Measure-Object -Maximum).Maximum
            $sum = 0
            
            for ($j = 0; $j -lt $cols; $j++) {
                $result[$i][$j] = [Math]::Exp($matrix[$i][$j] - $maxVal)
                $sum += $result[$i][$j]
            }
            
            for ($j = 0; $j -lt $cols; $j++) {
                $result[$i][$j] /= $sum
            }
        }
        return $result
    }

    static [double[][]] AddResidual([double[][]]$X, [double[][]]$Y) {
        [double[][]]$result = New-Object 'double[][]' $X.Count
        for ($i = 0; $i -lt $X.Count; $i++) {
            $result[$i] = New-Object 'double[]' $X[$i].Count
            for ($j = 0; $j -lt $X[$i].Count; $j++) {
                $result[$i][$j] = $X[$i][$j] + $Y[$i][$j]
            }
        }
        return $result
    }

    static [double[][]] LayerNorm([double[][]]$X) {
        $seq_len = $X.Count
        [double[][]]$result = New-Object 'double[][]' $seq_len
        $eps = 1e-6
        
        for ($i = 0; $i -lt $seq_len; $i++) {
            $mean = ($X[$i] | Measure-Object -Average).Average
            $variance = ($X[$i] | ForEach-Object { [Math]::Pow($_ - $mean, 2) } | Measure-Object -Average).Average
            $result[$i] = New-Object 'double[]' $X[$i].Count
            
            for ($j = 0; $j -lt $X[$i].Count; $j++) {
                $normalized = ($X[$i][$j] - $mean) / [Math]::Sqrt($variance + $eps)
                $result[$i][$j] = $normalized
            }
        }
        return $result
    }

    static [double[][]] Transpose([double[][]]$matrix) {
        $rows = $matrix.Count
        $cols = $matrix[0].Count
        [double[][]]$result = New-Object 'double[][]' $cols
        
        for ($i = 0; $i -lt $cols; $i++) {
            $result[$i] = New-Object 'double[]' $rows
            for ($j = 0; $j -lt $rows; $j++) {
                $result[$i][$j] = $matrix[$j][$i]
            }
        }
        return $result
    }
}

class SimpleAttention {
    [int]$embedding_dim
    [double[][]]$W_q
    [double[][]]$W_k
    [double[][]]$W_v
    [double[][]]$W_o

    [void] Initialize([int]$embedding_dim) {
        $this.embedding_dim = $embedding_dim
        $this.W_q = [MatrixOps]::CreateMatrix($embedding_dim, $embedding_dim, 0.01)
        $this.W_k = [MatrixOps]::CreateMatrix($embedding_dim, $embedding_dim, 0.01)
        $this.W_v = [MatrixOps]::CreateMatrix($embedding_dim, $embedding_dim, 0.01)
        $this.W_o = [MatrixOps]::CreateMatrix($embedding_dim, $embedding_dim, 0.01)
    }

    [double[][]] Forward([double[][]]$X) {
        [double]$scale = 1.0 / [Math]::Sqrt($this.embedding_dim)
        
        # Linear transformations
        [double[][]]$Q = [MatrixOps]::Multiply($X, $this.W_q, 1.0)
        [double[][]]$K = [MatrixOps]::Multiply($X, $this.W_k, 1.0)
        [double[][]]$V = [MatrixOps]::Multiply($X, $this.W_v, 1.0)
        
        # Compute attention scores
        [double[][]]$K_T = [MatrixOps]::Transpose($K)
        [double[][]]$scores = [MatrixOps]::Multiply($Q, $K_T, $scale)
        
        # Apply softmax
        [double[][]]$attn_weights = [MatrixOps]::Softmax($scores)
        
        # Apply to values
        [double[][]]$output = [MatrixOps]::Multiply($attn_weights, $V, 1.0)
        
        # Output projection
        $output = [MatrixOps]::Multiply($output, $this.W_o, 1.0)
        
        return $output
    }
}

class FeedForward {
    [int]$embedding_dim
    [int]$hidden_dim
    [double[][]]$W1
    [double[]]$b1
    [double[][]]$W2
    [double[]]$b2

    [void] Initialize([int]$embedding_dim, [int]$hidden_dim) {
        $this.embedding_dim = $embedding_dim
        $this.hidden_dim = $hidden_dim
        
        $this.W1 = [MatrixOps]::CreateMatrix($embedding_dim, $hidden_dim, 0.01)
        $this.b1 = New-Object 'double[]' $hidden_dim
        
        $this.W2 = [MatrixOps]::CreateMatrix($hidden_dim, $embedding_dim, 0.01)
        $this.b2 = New-Object 'double[]' $embedding_dim
    }

    [double[][]] Forward([double[][]]$X) {
        $seq_len = $X.Count
        [double[][]]$result = New-Object 'double[][]' $seq_len
        
        for ($i = 0; $i -lt $seq_len; $i++) {
            # First layer with ReLU
            [double[]]$hidden = New-Object 'double[]' $this.hidden_dim
            for ($j = 0; $j -lt $this.hidden_dim; $j++) {
                $sum = $this.b1[$j]
                for ($k = 0; $k -lt $this.embedding_dim; $k++) {
                    $sum += $X[$i][$k] * $this.W1[$k][$j]
                }
                $hidden[$j] = [Math]::Max(0, $sum)  # ReLU
            }
            
            # Second layer
            $result[$i] = New-Object 'double[]' $this.embedding_dim
            for ($j = 0; $j -lt $this.embedding_dim; $j++) {
                $sum = $this.b2[$j]
                for ($k = 0; $k -lt $this.hidden_dim; $k++) {
                    $sum += $hidden[$k] * $this.W2[$k][$j]
                }
                $result[$i][$j] = $sum
            }
        }
        return $result
    }
}

class TransformerBlock {
    [SimpleAttention]$attention
    [FeedForward]$feed_forward

    [void] Initialize([int]$embedding_dim, [int]$hidden_dim) {
        $this.attention = New-Object SimpleAttention
        $this.attention.Initialize($embedding_dim)
        
        $this.feed_forward = New-Object FeedForward
        $this.feed_forward.Initialize($embedding_dim, $hidden_dim)
    }

    [double[][]] Forward([double[][]]$X) {
        # Self-attention with residual
        [double[][]]$attn_output = $this.attention.Forward($X)
        [double[][]]$X_res = [MatrixOps]::AddResidual($X, $attn_output)
        $X_res = [MatrixOps]::LayerNorm($X_res)
        
        # Feed-forward with residual
        [double[][]]$ff_output = $this.feed_forward.Forward($X_res)
        [double[][]]$X_out = [MatrixOps]::AddResidual($X_res, $ff_output)
        $X_out = [MatrixOps]::LayerNorm($X_out)
        
        return $X_out
    }
}

class SmallLLM {
    [int]$vocab_size
    [int]$embedding_dim
    [int]$max_seq_len
    [double[][]]$embeddings
    [double[][]]$pos_encoding
    [TransformerBlock[]]$blocks
    [double[][]]$output_weight
    [double[]]$output_bias

    [void] Initialize([int]$vocab_size, [int]$embedding_dim, [int]$num_layers, [int]$hidden_dim, [int]$max_seq_len) {
        $this.vocab_size = $vocab_size
        $this.embedding_dim = $embedding_dim
        $this.max_seq_len = $max_seq_len
        
        # Token embedding
        $this.embeddings = [MatrixOps]::CreateMatrix($vocab_size, $embedding_dim, 0.01)
        
        # Positional encoding
        $this.pos_encoding = $this.CreatePositionalEncoding($max_seq_len, $embedding_dim)
        
        # Transformer blocks
        $this.blocks = @()
        for ($i = 0; $i -lt $num_layers; $i++) {
            $block = New-Object TransformerBlock
            $block.Initialize($embedding_dim, $hidden_dim)
            $this.blocks += $block
        }
        
        # Output layer
        $this.output_weight = [MatrixOps]::CreateMatrix($embedding_dim, $vocab_size, 0.01)
        $this.output_bias = New-Object 'double[]' $vocab_size
    }

    [double[][]] CreatePositionalEncoding([int]$seq_len, [int]$d_model) {
        [double[][]]$pos_enc = New-Object 'double[][]' $seq_len
        
        for ($pos = 0; $pos -lt $seq_len; $pos++) {
            $pos_enc[$pos] = New-Object 'double[]' $d_model
            
            for ($i = 0; $i -lt $d_model; $i += 2) {
                $denominator = [Math]::Pow(10000, $i / $d_model)
                $pos_enc[$pos][$i] = [Math]::Sin($pos / $denominator)
                
                if ($i + 1 -lt $d_model) {
                    $pos_enc[$pos][$i + 1] = [Math]::Cos($pos / $denominator)
                }
            }
        }
        return $pos_enc
    }

    [double[][]] Forward([int[]]$token_ids) {
        $seq_len = $token_ids.Count
        
        # Embedding
        [double[][]]$x = New-Object 'double[][]' $seq_len
        for ($i = 0; $i -lt $seq_len; $i++) {
            $x[$i] = $this.embeddings[$token_ids[$i]].Clone()
        }
        
        # Add positional encoding
        for ($i = 0; $i -lt $seq_len; $i++) {
            for ($j = 0; $j -lt $this.embedding_dim; $j++) {
                $x[$i][$j] += $this.pos_encoding[$i][$j]
            }
        }
        
        # Transformer blocks
        foreach ($block in $this.blocks) {
            $x = $block.Forward($x)
        }
        
        # Output projection
        [double[][]]$logits = [MatrixOps]::Multiply($x, $this.output_weight, 1.0)
        
        return $logits
    }

    [int[]] Generate([int[]]$prompt_ids, [int]$max_new_tokens, [double]$temperature) {
        [System.Collections.Generic.List[int]]$generated = [System.Collections.Generic.List[int]]::new()
        foreach ($id in $prompt_ids) {
            $generated.Add($id)
        }
        
        $random = [System.Random]::new()
        
        for ($t = 0; $t -lt $max_new_tokens; $t++) {
            # Get current sequence
            [int[]]$current_seq = if ($generated.Count -gt $this.max_seq_len) {
                $generated.GetRange($generated.Count - $this.max_seq_len, $this.max_seq_len).ToArray()
            } else {
                $generated.ToArray()
            }
            
            # Forward pass
            [double[][]]$logits = $this.Forward($current_seq)
            
            # Get logits for last token
            [double[]]$next_logits = $logits[$logits.Count - 1].Clone()
            
            # Apply temperature
            for ($i = 0; $i -lt $next_logits.Count; $i++) {
                $next_logits[$i] /= $temperature
            }
            
            # Softmax
            $max_logit = ($next_logits | Measure-Object -Maximum).Maximum
            [double[]]$probs = New-Object 'double[]' $next_logits.Count
            $sum = 0
            
            for ($i = 0; $i -lt $next_logits.Count; $i++) {
                $probs[$i] = [Math]::Exp($next_logits[$i] - $max_logit)
                $sum += $probs[$i]
            }
            
            for ($i = 0; $i -lt $probs.Count; $i++) {
                $probs[$i] /= $sum
            }
            
            # Sample next token
            $randVal = $random.NextDouble()
            $cumulative = 0
            $next_token = 0
            
            for ($i = 0; $i -lt $probs.Count; $i++) {
                $cumulative += $probs[$i]
                if ($randVal -le $cumulative) {
                    $next_token = $i
                    break
                }
            }
            
            $generated.Add($next_token)
        }
        
        return $generated.ToArray()
    }

    [double] ComputeLoss([int[]]$token_ids) {
        [double[][]]$logits = $this.Forward($token_ids)
        
        # Shift logits and targets
        [double[][]]$logits_shifted = $logits[0..($logits.Count - 2)]
        [int[]]$targets = $token_ids[1..($token_ids.Count - 1)]
        
        # Cross-entropy loss
        $total_loss = 0
        $count = 0
        
        for ($idx = 0; $idx -lt $targets.Count; $idx++) {
            $max_logit = ($logits_shifted[$idx] | Measure-Object -Maximum).Maximum
            $exp_sum = 0
            
            for ($i = 0; $i -lt $logits_shifted[$idx].Count; $i++) {
                $exp_sum += [Math]::Exp($logits_shifted[$idx][$i] - $max_logit)
            }
            
            $prob = [Math]::Exp($logits_shifted[$idx][$targets[$idx]] - $max_logit) / $exp_sum
            $total_loss += -[Math]::Log($prob + 1e-10)
            $count++
        }
        
        return $total_loss / $count
    }
}

class SimpleTokenizer {
    [char[]]$chars
    [hashtable]$char_to_id
    [hashtable]$id_to_char
    [int]$vocab_size

    [void] Initialize([string]$text) {
        $this.chars = $text.ToCharArray() | Select-Object -Unique | Sort-Object
        $this.char_to_id = @{}
        $this.id_to_char = @{}
        $this.vocab_size = $this.chars.Count
        
        for ($i = 0; $i -lt $this.chars.Count; $i++) {
            $this.char_to_id[[string]$this.chars[$i]] = $i
            $this.id_to_char[$i] = $this.chars[$i]
        }
    }

    [int[]] Encode([string]$text) {
        [System.Collections.Generic.List[int]]$encoded = [System.Collections.Generic.List[int]]::new()
        foreach ($char in $text.ToCharArray()) {
            $charStr = [string]$char
            if ($this.char_to_id.ContainsKey($charStr)) {
                $encoded.Add($this.char_to_id[$charStr])
            }
        }
        return $encoded.ToArray()
    }

    [string] Decode([int[]]$token_ids) {
        [System.Text.StringBuilder]$sb = [System.Text.StringBuilder]::new()
        foreach ($id in $token_ids) {
            $sb.Append($this.id_to_char[$id]) | Out-Null
        }
        return $sb.ToString()
    }
}

# Example Usage
Write-Host "=== Small Language Model in PowerShell ===" -ForegroundColor Cyan
Write-Host ""

# Create sample dataset
$sample_text = "hello world, this is a small language model! " * 5

# Initialize tokenizer
$tokenizer = New-Object SimpleTokenizer
$tokenizer.Initialize($sample_text)
Write-Host "Vocabulary size: $($tokenizer.vocab_size)" -ForegroundColor Green
Write-Host "Characters: $($tokenizer.chars -join ', ')" -ForegroundColor Green
Write-Host ""

# Create model
Write-Host "Creating model..." -ForegroundColor Yellow
$model = New-Object SmallLLM
$model.Initialize(
    $tokenizer.vocab_size,    # vocab_size
    32,                         # embedding_dim
    2,                          # num_layers
    128,                        # hidden_dim
    64                          # max_seq_len
)
Write-Host "Model created successfully!" -ForegroundColor Green
Write-Host ""

# Encode sample text
[int[]]$token_ids = $tokenizer.Encode($sample_text)
Write-Host "Encoded sequence length: $($token_ids.Count)" -ForegroundColor Green
Write-Host ""

# Compute loss on a small sequence
Write-Host "Computing loss on test sequence..." -ForegroundColor Yellow
if ($token_ids.Count -gt 5) {
    [int[]]$test_seq = $token_ids[0..19]
    [double]$loss = $model.ComputeLoss($test_seq)
    Write-Host "Initial loss: $([Math]::Round($loss, 4))" -ForegroundColor Green
} else {
    Write-Host "Sequence too short for loss computation" -ForegroundColor Yellow
}
Write-Host ""

# Generate text from a prompt
$prompt = "hello"
[int[]]$prompt_ids = $tokenizer.Encode($prompt)
Write-Host "Prompt: '$prompt'" -ForegroundColor Cyan

Write-Host "Generating text..." -ForegroundColor Yellow
[int[]]$generated_ids = $model.Generate($prompt_ids, 30, 0.8)
[string]$generated_text = $tokenizer.Decode($generated_ids)
Write-Host "Generated: '$generated_text'" -ForegroundColor Green
Write-Host ""

Write-Host "=== Generation Complete ===" -ForegroundColor Cyan
