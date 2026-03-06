#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : KalmanFilterTracker.py
#   Author      : Yu Bai, Chunliu Xu, Chengcheng Hou
#   Created date: 2020-07-22
#   Description :
#
#================================================================

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
        self.dt = 0.005  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.b = self.lastResult
        else:  # update using detection
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                              self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u




class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_length: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def _create_track(self, detection):
        self.tracks.append(Track(detection, self.trackIdCount))
        self.trackIdCount += 1

    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost matrix (Euclidean distance) between
              predicted track positions and detected centroids
            - Using Hungarian Algorithm assign detections to tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Reject assignments exceeding the distance threshold
            - Remove tracks missing for too long
            - Start new tracks for unmatched detections
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Initialize tracks from first batch of detections
        if not self.tracks:
            for det in detections:
                self._create_track(det)

        N, M = len(self.tracks), len(detections)

        # Build Euclidean distance cost matrix (N tracks x M detections)
        cost = np.zeros((N, M))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                diff = track.prediction - det
                cost[i, j] = np.linalg.norm(diff)

        # Hungarian algorithm: assign detections to tracks
        assignment = [-1] * N
        for r, c in zip(*linear_sum_assignment(cost)):
            assignment[r] = c

        # Reject assignments that exceed the distance threshold;
        # increment skipped_frames for all unmatched tracks
        for i, j in enumerate(assignment):
            if j == -1 or cost[i, j] > self.dist_thresh:
                assignment[i] = -1
                self.tracks[i].skipped_frames += 1

        # Remove stale tracks — iterate in reverse to preserve indices
        stale_indices = sorted(
            [i for i, t in enumerate(self.tracks)
             if t.skipped_frames > self.max_frames_to_skip],
            reverse=True,
        )
        for i in stale_indices:
            del self.tracks[i]
            del assignment[i]

        # Start new tracks for detections not matched to any existing track
        matched_dets = set(assignment)
        for j in range(M):
            if j not in matched_dets:
                self._create_track(detections[j])

        # Update Kalman filter state, trace, and lastResult for each track
        for i, track in enumerate(self.tracks):
            track.KF.predict()

            if assignment[i] != -1:
                track.skipped_frames = 0
                track.prediction = track.KF.correct(detections[assignment[i]], 1)
            else:
                track.prediction = track.KF.correct(np.array([[0], [0]]), 0)

            track.trace.append(track.prediction)
            if len(track.trace) > self.max_trace_length:
                track.trace = track.trace[-self.max_trace_length:]

            track.KF.lastResult = track.prediction




