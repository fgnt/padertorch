import numpy as np
import paderbox as pb
import padertorch as pt
from einops import einops
import torch
from scipy.spatial.distance import cosine

from itertools import permutations, chain
from functools import partial

from lazy_dataset import FilterException
from padertorch.contrib.cb.summary import ReviewSummary
from padertorch.contrib.tcl.speaker_embeddings.dvectors import ResNet34
from padertorch.contrib.tcl.speaker_embeddings.student_embeddings import StudentdVectors
from padertorch.contrib.tcl.speaker_embeddings.eer_metrics import get_eer, get_dcf
from padertorch.contrib.je.modules.reduce import Mean


class TeacherStudentEmbeddings(pt.Model):
    def __init__(self,
                 teacher=None,
                 student=None,
                 silence_masking=False,
                 loss_masking=False,
                 sample_rate=16000,
                 num_spk=2,
                 pit_order='utterance',  # ['frame', 'utterance', None]
                 framewise_loss_fn='mse',
                 use_framewise_loss=True,
                 use_embedding_loss=False,
                 use_geodesic_loss=True,
                 aggregate='mean',
                 normalize=True,
                 teacher_embedding_key='dvector',
                 ):
        """
        Parent Model for the combination of a d-vector teacher and a frame-wise student model.
        Used for the training of teacher-student embeddings in:

        [1] T. Cord-Landwehr, C. Boeddeker, C. Zorilă, R. Doddipatla and R. Haeb-Umbach,
        "Frame-Wise and Overlap-Robust Speaker Embeddings for Meeting Diarization,"
        ICASSP 2023, doi: 10.1109/ICASSP49357.2023.10095370.

        [2]Cord-Landwehr, T., Boeddeker, C., Zorilă, C., Doddipatla, R., & Haeb-Umbach, R. (2023).
        "A Teacher-Student approach for extracting informative speaker embeddings from speech mixtures"
         arXiv preprint arXiv:2306.00634.
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss_masking = loss_masking
        self.silence_masking = silence_masking
        self.sampling_rate = sample_rate
        self.num_spk = num_spk
        self.teacher_signal = 'speech_image'
        self.use_framewise_loss = use_framewise_loss
        self.use_embedding_loss = use_embedding_loss
        self.normalize = normalize
        self.cos = torch.nn.CosineSimilarity()
        self.pit_order = pit_order
        self.aggregate = aggregate
        if framewise_loss_fn == 'mse':
            self.loss_fn = partial(pt.mse_loss, reduction=None)
        elif framewise_loss_fn == 'log_mse':
            self.loss_fn = partial(pt.log_mse_loss, reduction=None)
        elif framewise_loss_fn == 'cosine':
            self.loss_fn = self.cosine_loss
        elif framewise_loss_fn == 'log_cosine':
            self.loss_fn = self.log_cosine_loss
        else:
            self.d_vector_loss_fn = self.cosine_loss
        self.geodesic_loss = use_geodesic_loss
        self.reduction = 4
        self.target = teacher_embedding_key
        self.softmax_ce = torch.nn.CrossEntropyLoss()


    def cosine_loss(self, x, y):
        return 1 - self.cos(x, y)

    def log_cosine_loss(self, x, y):
        return torch.log10(1 - self.cos(x, y))

    def embedding_loss_fn(self, x, y):
        return torch.mean(self.loss_fn(x, y))


    @classmethod
    def finalize_dogmatic_config(cls, config):
        """Provides a default configuration. See `padertorch.configurable` for
        details."""
        config['student'] = {'factory': StudentdVectors}
        config['teacher'] = {'factory': ResNet34}


    def get_teacher_embeddings(self, example):
        """
        Call the teacher module to obtain a d-vector for each clean speech source present in the observation

        Returns:
            teacher_d_vectors: Time-averaged teacher d-vectors, shape (B K E)
            teacher_embeddings: Teacher embeddings before time-averaging, shape (B K E T)
        """
        with torch.no_grad():
            if isinstance(example['features_teacher'], list):
                features_teacher = list(chain.from_iterable(example['features_teacher']))
                teacher_length = torch.tensor(list(chain.from_iterable(example['num_frames_teacher'])))

                features_teacher = pt.pad_sequence(features_teacher, batch_first=True)

                features_teacher = einops.rearrange(features_teacher, '(b k) t f-> (b k) t f', k=self.num_spk)
            else:
                teacher_length = torch.tensor(example['num_frames_teacher'])
                features_teacher = example['features_teacher']
                if teacher_length.ndim < 2:
                    teacher_length = einops.repeat(teacher_length, 'b -> b k', k=self.num_spk)
                teacher_length = einops.rearrange(teacher_length, 'b k -> (b k)')
                features_teacher = pt.pad_sequence(features_teacher, batch_first=True)
                features_teacher = einops.rearrange(features_teacher, 'b k t f-> (b k) t f')
            teacher_d_vectors, teacher_embeddings = self.teacher(features_teacher, teacher_length)
            teacher_d_vectors = einops.rearrange(teacher_d_vectors, '(b k) e -> b k e', k=self.num_spk)
            teacher_embeddings = einops.rearrange(teacher_embeddings, '(b k) e t-> b k e t', k=self.num_spk)

        return teacher_d_vectors, teacher_embeddings

    def compute_geodesic_loss(self, embeddings, targets, ov_boundaries, single_speaker_targets):
        """
        Solve Constrained least squares problem to find optimal point between both target embeddings
        """

        ov_boundaries = [ov//self.reduction for ov in ov_boundaries]
        loss_spk1 = self.loss_fn(embeddings[0, :, :ov_boundaries[0]].transpose(0, 1),
                                 targets[0, :, :ov_boundaries[0]].transpose(0, 1))
        loss_spk2 = self.loss_fn(embeddings[0, :, ov_boundaries[1]:].transpose(0, 1),
                                 targets[1, :, ov_boundaries[1]:].transpose(0, 1))
        try:
            alpha = torch.einsum('kt,Kk->Kt', torch.einsum('ke,et->kt', single_speaker_targets,
                                                           embeddings[0, :, ov_boundaries[0]:ov_boundaries[1]]),
                                 torch.einsum('Ke,ke->kK', single_speaker_targets, single_speaker_targets).inverse())
            w = torch.einsum('ke,Ke->kK', single_speaker_targets, single_speaker_targets).inverse()
            alpha = alpha - torch.einsum('k,t->kt', torch.sum(w, dim=-1) * (1 / torch.sum(w)),
                                         torch.sum(alpha, dim=0) - 1)
        except:
            alpha = torch.tensor((0.5, 0.5), device=embeddings.device)
            alpha = einops.repeat(alpha, 'k -> k t', t=ov_boundaries[1] - ov_boundaries[0])
        ov_target = torch.einsum('ke,kt->et', single_speaker_targets, alpha)
        ov_target = ov_target / torch.std(ov_target, dim=0, keepdim=True).detach()
        ov_target = ov_target * torch.std(single_speaker_targets[0])

        loss_ov = self.loss_fn(embeddings[0, :, ov_boundaries[0]:ov_boundaries[1]].transpose(0, 1),
                               ov_target.transpose(0, 1)
                               )
        geodesic_loss = torch.cat((loss_spk1, loss_spk2, loss_ov))
        return geodesic_loss, alpha

    def forward(self, example):
        sequence_lengths = example['num_frames_student']
        sequence = pt.pad_sequence(example['features_student'], batch_first=True)
        sequence = einops.rearrange(sequence, 'b t f -> b f t')
        student_embeddings, sequence_lengths = self.student(sequence, sequence_lengths)
        return student_embeddings, sequence_lengths

    def review(self, example, outputs):
        embeddings, seq_lens = outputs
        summary = ReviewSummary(sampling_rate=self.sampling_rate)
        B, K, E, T = embeddings.shape

        # obtain teacher embeddings
        teacher_d_vectors, teacher_embeddings = self.get_teacher_embeddings(example)

        teacher_d_vectors = torch.split(teacher_d_vectors, example['num_speaker'])

        embeddings_weights = []
        framewise_loss = []
        utterance_loss = []
        assert self.use_framewise_loss or self.use_embedding_loss,  'At least one loss needs to be active!'
        embeddings_reordered = []
        d_vectors = []
        if K > 1:
            similarities = {'same_speaker': [],
                            'different_speaker': []}

        for b, seq_len in enumerate(seq_lens):
            if self.target == 'dvector':
                target_embeddings = einops.repeat(teacher_d_vectors[b][:, 0, :], 'k e -> k e t', t=max(seq_lens))
            elif self.target == 'embedding':
                target_embeddings = teacher_embeddings[b]
            else:
                raise NotImplementedError
            # Compute framewise metrics
            if self.use_framewise_loss:
                if self.geodesic_loss and target_embeddings.shape[0] > 1:
                    l, weights = self.compute_geodesic_loss(embeddings[b], target_embeddings, example['overlap_boundaries'][b],
                                                   teacher_d_vectors[b][:, 0, :])
                    weights = pt.utils.to_numpy(weights, detach=True)
                    embeddings_weights.append(np.sort(weights, axis=0))
                    embedding = embeddings[b]
                elif self.pit_order == 'frame':
                    l, embedding = self.framewise_reconstruction_loss(embeddings[b], target_embeddings)
                else:
                    l, p = self.utterance_reconstruction_loss(embeddings[b], target_embeddings)
                    if self.pit_order == 'utterance':
                        embedding = embeddings[b, p, ...]
                    else:
                        embedding = embeddings[b]
                framewise_loss.append(l)
            else:
                embedding = embeddings[b]
            embeddings_reordered.append(embedding)

            # Compute utterance-wise metrics
            if self.aggregate == 'mean':
                # Calculate Mean over reduced time dim (different len for each example)
                d_vector = Mean(axis=-1)(embedding)
            else:
                d_vector = embedding

            if self.normalize:
                d_vector = d_vector / torch.norm(d_vector, dim=-1, keepdim=True)
            d_vectors.append(d_vector)

            utterance_loss.append(self.d_vector_loss_fn(d_vector, teacher_d_vectors[b][:, 0, :]))

            if K > 1:
                student = pt.utils.to_numpy(d_vector, detach=True)
                teacher = pt.utils.to_numpy(teacher_d_vectors[b][:,0,:], detach=True)

                [similarities['same_speaker'].append(1 - cosine(teacher[k, :], student[k, :]))
                 for k in range(self.num_spk)]
                [similarities['different_speaker'].append(1 - cosine(teacher[k - 1, :], student[k, :]))
                 for k in range(self.num_spk)]

        embedding_norm = einops.rearrange(torch.norm(torch.stack(embeddings_reordered), dim=-2), 'b k t -> (b k) t')

        utterance_loss = torch.mean(torch.cat(utterance_loss))
        if self.use_utterance_loss:
            summary.add_to_loss(utterance_loss)
        summary.add_scalar('d_vector_loss', pt.utils.to_numpy(utterance_loss, detach=True))

        if self.use_framewise_loss:
            framewise_loss = torch.mean(torch.stack(framewise_loss))
            summary.add_to_loss(framewise_loss)
        summary.add_scalar('frame_level_loss', pt.utils.to_numpy(framewise_loss, detach=True))
        summary.add_scalar('_embedding_norm', torch.mean(embedding_norm))

        if len(embeddings_weights) > 0:
            embeddings_weights = np.concatenate(embeddings_weights, axis=-1)
            summary.add_histogram('geodesic_weight_1', np.mean(embeddings_weights[0]))
            summary.add_histogram('geodesic_weight_2', np.mean(embeddings_weights[1]))

        if K > 1:
            # Track cross-speaker similarities for more than one output channel
            summary.add_histogram('same_speaker_similarities', similarities['same_speaker'])
            summary.add_histogram('different_speaker_similarities', similarities['different_speaker'])
            summary.add_scalar('same_speaker_score', np.mean(np.stack(similarities['same_speaker'])))
            summary.add_scalar('different_speaker_score', np.mean(np.stack(similarities['different_speaker'])))
        if self.create_snapshot:
            summary.add_audio('observation', example['observation'][0])
            #summary.add_image()
            summary.add_spectrogram_image('features_student', torch.exp(example['features_student'][0]))
            for k in range(self.num_spk):
                summary.add_spectrogram_image(f'features_teacher_{k}', torch.exp(example['features_teacher'][0][k]))

        if not self.training:
            # Ensure every embedding has length of 1

            if len(d_vectors) > 0:
                d_vectors = torch.stack(d_vectors)
            elif len(embeddings_reordered) > 0:
                embeddings_reordered = torch.stack(embeddings_reordered)
                d_vectors = torch.mean(embeddings_reordered, dim=-1)
            d_vectors = d_vectors #/ (torch.norm(d_vectors, dim=1, keepdim=True) + 1e-8)
            summary.add_buffer('embeddings', d_vectors)
            summary.add_buffer('example_ids', example['example_id'])
            summary.add_buffer('speaker_ids', example['speaker_id'])

        return summary

    def update_activity(self, activity):
        activity = activity[:, ::self.reduction]
        return activity

    def utterance_reconstruction_loss(self, estimates, targets):
        """
        """
        K, E, T = estimates.shape
        assert estimates.shape == targets.shape, (estimates.shape, targets.shape)
        if K > 1:
            loss, perm = pt.pit_loss(estimates, targets, axis=0, return_permutation=True,
                                     loss_fn=self.embedding_loss_fn)
        else:
            loss = pt.pit_loss(estimates, targets, axis=0, return_permutation=False,
                               loss_fn=self.embedding_loss_fn)
            perm = (0, )
        return loss, perm

    def framewise_reconstruction_loss(self, estimates, targets):
        K, E, T = estimates.shape
        assert estimates.shape == targets.shape, (estimates.shape, targets.shape)
        embedding = einops.rearrange(estimates, 'speaker embedding time -> (time speaker) embedding')
        targets = einops.rearrange(targets, 'speaker embedding time -> time speaker embedding')
        loss_matrix = []
        p_idx_to_perm = []
        for p in permutations(range(K), K):
            p_idx_to_perm.append(torch.tensor(p))
            _target = targets[:, p, :]
            _target = einops.rearrange(_target, 'n speaker  embedding-> (n speaker) embedding')
            _loss = self.loss_fn(embedding, _target)
            _loss = einops.rearrange(_loss, '(time speaker) -> time speaker', time=T, speaker=K)
            _loss = einops.reduce(_loss, 'time speaker -> time', 'sum', time=T)
            loss_matrix.append(_loss)

        p_idx_to_perm = torch.stack(p_idx_to_perm).to(device=estimates.device)
        loss_matrix = torch.stack(loss_matrix)
        speaker_loss, best_perm = torch.min(loss_matrix, dim=0)
        perm_indexer = p_idx_to_perm[best_perm]
        perm_indexer = einops.rearrange(perm_indexer, 'time speaker -> time speaker', time=T)
        perm_indexer = einops.repeat(
            perm_indexer, 'time speaker -> time speaker e', e=E
        ).to(embedding.device, dtype=torch.long)
        embedding = einops.rearrange(embedding, '(time speaker) e -> time speaker e', time=T, speaker=K)
        embeddings_reordered = torch.gather(embedding, -2, perm_indexer)
        embeddings_reordered = einops.rearrange(embeddings_reordered, ' t k e -> k e t')
        return speaker_loss, embeddings_reordered

    def modify_summary(self, summary):
        if 'embeddings' in summary['buffers']:

            embeddings = sum(summary['buffers']['embeddings'], [])

            embeddings = np.concatenate(embeddings, axis=0)
            # Subtract global mean to obtain more meaningful cosine distance
            #validation_mean = np.mean(embeddings, axis=0, keepdims=True)
            embeddings = embeddings #- validation_mean
            examples = sum(summary['buffers']['example_ids'], [])
            speakers = sum(summary['buffers']['speaker_ids'], [])
            print('Obtaining metrics for the validation step')

            indexer = list(range(len(examples)))
            np.random.default_rng(42).shuffle(indexer)
            scores = list()
            labels = list()
            for idx1, idx2 in enumerate(indexer):
                if self.num_spk == 1:
                    e1 = embeddings[idx1]
                    e2 = embeddings[idx2]
                    s1 = speakers[idx1]
                    s2 = speakers[idx2]
                    if len(e1.shape) > 1:
                        e1 = e1[0, :]
                        e2 = e2[0, :]
                        s1 = s1[0]
                        s2 = s2[0]
                    labels.append(s1 == s2)
                    scores.append(1 - cosine(e1, e2))
                else:
                    multispeaker_labels, multispeaker_scores = multispeaker_verification(
                        (embeddings[idx1], speakers[idx1]), (embeddings[idx2], speakers[idx2]))
                    [labels.append(label) for label in multispeaker_labels]
                    [scores.append(score) for score in multispeaker_scores]

            eer = get_eer(np.array(scores), np.array(labels))
            dcf = get_dcf(np.array(scores), np.array(labels))
            print(f'(Pseudo) Equal error rate for validation is: {eer}')
            summary['scalars']['EER'] = eer
            summary['scalars']['minDCF'] = dcf
            summary['histograms']['scores'] = np.array(scores)
            summary['histograms']['score_distance'] = np.abs(np.array(labels) - np.array(scores))
            summary['buffers'].clear()
        return super().modify_summary(summary)


def multispeaker_verification(example1, example2):
    e1, s1 = example1
    s1 = s1[0]
    e2, s2 = example2
    s2 = s2[0]

    K = len(s1)
    labels = []
    scores = []
    for idx1 in range(K):
        for idx2 in range(K):
            labels.append(s1[idx1] == s2[idx2])
            scores.append(1 - cosine(e1[idx1], e2[idx2]))
    return labels, scores
