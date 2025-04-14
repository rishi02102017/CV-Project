import numpy as np

class LaneTokenizer:
    def __init__(self, nbins=1000):
        self.nbins = nbins

        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.FORMAT_TOKENS = {
            'segmentation': 3,
            'anchor': 4,
            'parameter': 5
        }
        self.LANE_TOKEN = 6
        
        # Update vocab size
        self.vocab_size = self.nbins + 7  # nbins + pad + special tokens

    def normalize_and_quantize(self, x, y, width, height):
        x_norm = min(int((x / width) * (self.nbins - 1)), self.nbins - 1)
        y_norm = min(int((y / height) * (self.nbins - 1)), self.nbins - 1)
        return x_norm, y_norm

    def encode(self, annotation, image_size, format_type='anchor'):
        #print(f"[DEBUG] Image size passed to tokenizer: {image_size}")
        width, height = image_size

        input_seq = [self.START_TOKEN, self.FORMAT_TOKENS[format_type]]
        target_seq = [self.FORMAT_TOKENS[format_type]]

        
        #print(f"[DEBUG] Width: {width}, Height: {height}")


        if format_type in ['segmentation', 'anchor']:
            lanes = annotation['lanes']
            for lane in lanes:
                for point in lane['points']:
                    x_token, y_token = self.normalize_and_quantize(point[0], point[1], width, height)
                    input_seq.extend([x_token, y_token])
                    target_seq.extend([x_token, y_token])
                input_seq.append(self.LANE_TOKEN)
                target_seq.append(self.LANE_TOKEN)

        elif format_type == 'parameter':
            lanes = annotation['lanes']
            for lane in lanes:
                params = lane['params']  # Already normalized
                offset = lane['offset']

                param_tokens = [min(int(self.nbins * self.sigmoid(p)), self.nbins - 1) for p in params]
                offset_token = min(int((offset / height) * (self.nbins - 1)), self.nbins - 1)

                input_seq.extend(param_tokens + [offset_token, self.LANE_TOKEN])
                target_seq.extend(param_tokens + [offset_token, self.LANE_TOKEN])

        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        input_seq.append(self.END_TOKEN)
        target_seq.append(self.END_TOKEN)

        # ✅ Optional Safety Checks: Ensure tokens are in range
        assert all(0 <= t < self.vocab_size for t in input_seq), f"Input seq token out of range! Got max {max(input_seq)}"
        assert all(0 <= t < self.vocab_size for t in target_seq), f"Target seq token out of range! Got max {max(target_seq)}"

        return input_seq, target_seq

    def decode(self, sequence, image_size, format_type='anchor'):
        width, height = image_size

        idx = 1  # Skip START_TOKEN
        assert sequence[idx] == self.FORMAT_TOKENS[format_type]
        idx += 1

        lanes = []

        print(f"[DEBUG] Image size passed to tokenizer: {image_size}")
        print(f"[DEBUG] Width: {width}, Height: {height}")


        if format_type in ['segmentation', 'anchor']:
            current_lane = []
            while idx < len(sequence):
                token = sequence[idx]

                if token == self.LANE_TOKEN:
                    if current_lane:
                        lanes.append({'points': current_lane})
                        current_lane = []
                    idx += 1
                    continue
                elif token == self.END_TOKEN:
                    break

                x_token = sequence[idx]
                y_token = sequence[idx + 1]
                x = (x_token / (self.nbins - 1)) * width
                y = (y_token / (self.nbins - 1)) * height
                current_lane.append([x, y])
                idx += 2

        elif format_type == 'parameter':
            while idx < len(sequence):
                token = sequence[idx]

                if token == self.END_TOKEN:
                    break

                params = []
                for _ in range(5):
                    param_token = sequence[idx]
                    value = self.inverse_sigmoid(param_token / self.nbins)
                    params.append(value)
                    idx += 1

                offset_token = sequence[idx]
                offset = (offset_token / (self.nbins - 1)) * height
                idx += 1

                lanes.append({'params': params, 'offset': offset})

                if idx < len(sequence) and sequence[idx] == self.LANE_TOKEN:
                    idx += 1

        return lanes

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def inverse_sigmoid(y):
        y = np.clip(y, 1e-7, 1 - 1e-7)
        return -np.log((1 / y) - 1)
