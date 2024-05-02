from torch.nn.utils.rnn import pad_sequence
# unused at the moment
def custom_collate_fn(batch):
    """
    Custom collate function for handling variable-length sequences.
    Pads features (images) and stacks other tensors.
    """
    # print(f'batch={batch}')
    # Unzip the batch into separate tensors
    features, labels, bboxes, num_objects = zip(*batch)
    # print(f'features={features}')
    # print(f'labels={labels}')
    # print(f'bboxes={bboxes}')
    # print(f'num_objects={num_objects}')

    
    # Determine the maximum label length
    max_label_length = max(len(label) for label in labels)

    # Pad labels to the maximum length
    padded_labels = []
    for label in labels:
        pad_length = max_label_length - len(label)
        padded_label = torch.cat([label, torch.zeros(pad_length, dtype=label.dtype)])
        padded_labels.append(padded_label)

    # # Determine the maximum bboxes length
    # max_bboxes_length = max(len(bbox) for bbox in bboxes)

    # # Pad bboxes to the maximum length
    # padded_bboxes = []
    # for bbox in bboxes:
    #     pad_length = max_bboxes_length - len(bbox)
    #     padded_bbox = torch.cat([bbox, torch.zeros(pad_length, dtype=bbox.dtype)])
    #     padded_bboxes.append(padded_bbox)
    
    # Determine the maximum bbox count (number of bounding boxes) in the batch
    max_bbox_count = max(len(bbox) for bbox in bboxes)

    # Pad bboxes to the maximum count
    padded_bboxes = []
    for bbox in bboxes:
        pad_count = max_bbox_count - len(bbox)
        padded_bbox = torch.cat([bbox, torch.zeros(pad_count, 4)])  # Assuming each bbox is represented as [x1, y1, x2, y2]
        padded_bboxes.append(padded_bbox)

    # Pad the features (assuming they are 2D tensors)
    features_padded = pad_sequence(features, batch_first=True)

    # Stack other tensors
    labels_stacked = torch.stack(padded_labels)
    bboxes_stacked = torch.stack(padded_bboxes)
    num_objects_stacked = torch.stack(num_objects)

    return features_padded, labels_stacked, bboxes_stacked, num_objects_stacked