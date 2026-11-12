#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Scene descriptors, each checked against a scene whose answer is known by construction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stabilo_benchmark.masking import Labels, corners_to_hbb, empty_labels
from stabilo_benchmark.scene import (
    anchored_motion,
    background_texture,
    box_dimensions,
    coverage,
    describe,
    object_patches,
    platform_motion,
    summarize_fates,
)

WIDTH, HEIGHT = 640, 480


def boxes(*rects) -> Labels:
    """
    Labels from (x, y, w, h) rectangles, as axis-aligned oriented boxes.

    `hbb` is filled from the corners, as `load_labels` does.
    """
    obb = np.array([[x, y, x + w, y, x + w, y + h, x, y + h] for x, y, w, h in rects], dtype=np.float64).reshape(-1, 8)
    return Labels(classes=np.zeros(len(rects), dtype=int), hbb=corners_to_hbb(obb), obb=obb)


def textured(width: int = WIDTH, height: int = HEIGHT, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = cv2.GaussianBlur(rng.integers(0, 255, (height, width), dtype=np.uint8), (5, 5), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class TestCoverage:
    def test_a_single_box_covers_its_own_area(self):
        # Rasterization includes both edges, so a w x h box lands on (w+1) x (h+1) pixels.
        assert coverage(boxes((0, 0, 64, 48)), WIDTH, HEIGHT) == pytest.approx(65 * 49 / (WIDTH * HEIGHT))

    def test_overlap_is_counted_once(self):
        overlapping = coverage(boxes((0, 0, 64, 48), (32, 0, 64, 48)), WIDTH, HEIGHT)
        assert overlapping == pytest.approx(97 * 49 / (WIDTH * HEIGHT))
        assert overlapping < 2 * 65 * 49 / (WIDTH * HEIGHT)

    def test_no_objects_is_zero(self):
        assert coverage(empty_labels(), WIDTH, HEIGHT) == 0.0

    def test_degenerate_frame_is_nan(self):
        assert np.isnan(coverage(boxes((0, 0, 10, 10)), 0, 0))


class TestBackgroundTexture:
    def test_a_flat_frame_has_none(self):
        flat = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        assert background_texture(flat, empty_labels()) == pytest.approx(0.0)

    def test_a_textured_frame_has_some(self):
        assert background_texture(textured(), empty_labels()) > 100

    def test_it_is_comparable_across_resolutions(self):
        # Texture is measured at a fixed working resolution.
        small = textured(640, 480, seed=3)
        large = cv2.resize(small, (1280, 960), interpolation=cv2.INTER_CUBIC)
        a, b = background_texture(small, empty_labels()), background_texture(large, empty_labels())
        assert a == pytest.approx(b, rel=0.35)

    def test_object_pixels_are_excluded(self):
        frame = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        frame[100:200, 100:300] = textured(200, 100, seed=1)
        without = background_texture(frame, empty_labels())
        with_mask = background_texture(frame, boxes((100, 100, 200, 100)))
        assert without > 0
        assert with_mask == pytest.approx(0.0)


class TestDescribe:
    def test_describe_returns_every_descriptor(self):
        stats = describe(textured(), boxes((100, 100, 64, 48)), WIDTH, HEIGHT)
        assert stats.n_objects == 1
        assert 0.0 < stats.coverage < 1.0
        assert np.isfinite(stats.texture)

    def test_texture_is_skipped_without_an_image(self):
        assert np.isnan(describe(None, boxes((100, 100, 64, 48)), WIDTH, HEIGHT).texture)


def rotated(cx, cy, length, width, angle_deg):
    """Corners (4, 2) of a length x width box centered at (cx, cy), long side at `angle_deg`."""
    a = np.deg2rad(angle_deg)
    u, v = np.array([np.cos(a), np.sin(a)]) * length / 2, np.array([-np.sin(a), np.cos(a)]) * width / 2
    c = np.array([cx, cy])
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def oriented(quads) -> Labels:
    obb = np.asarray(quads, dtype=np.float64).reshape(-1, 8)
    return Labels(classes=np.zeros(len(obb), dtype=int), hbb=corners_to_hbb(obb), obb=obb)


class World:
    """
    A textured ground with distinct textured objects, filmed twice: the reference with the identity,
    the later frame through a known platform motion (an affine map of the whole frame).
    """

    SIZE = (540, 960)
    CAR = (52, 24)

    def __init__(self, count=30, seed=7, scale=1.0):
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        height, width = (round(d * scale) for d in self.SIZE)
        self.shape = (height, width)
        noise = self.rng.integers(40, 215, (height // 4 + 1, width // 4 + 1), dtype=np.uint8)
        ground = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        self.ground = cv2.cvtColor(cv2.GaussianBlur(ground, (0, 0), 1.5 * scale), cv2.COLOR_GRAY2BGR)
        self.looks = [self._look() for _ in range(count)]
        margin = 120 * scale
        self.places = [
            (float(self.rng.uniform(margin, width - margin)), float(self.rng.uniform(margin, height - margin)))
            for _ in range(count)
        ]
        self.angles = [float(self.rng.choice([0.0, 90.0, 30.0, -45.0])) for _ in range(count)]

    def _look(self):
        length, width = (round(d * self.scale) for d in self.CAR)
        coarse = self.rng.integers(0, 255, (max(2, width // 6), max(2, length // 6), 3), dtype=np.uint8)
        return cv2.resize(coarse, (length, width), interpolation=cv2.INTER_NEAREST)

    def _paint(self, image, look, quad):
        length, width = look.shape[1], look.shape[0]
        source = np.float32([[0, 0], [length - 1, 0], [length - 1, width - 1]])
        warp = cv2.getAffineTransform(source, np.float32(quad[:3]))
        layer = cv2.warpAffine(look, warp, image.shape[1::-1])
        inside = cv2.warpAffine(np.full(look.shape[:2], 255, np.uint8), warp, image.shape[1::-1]) > 0
        image[inside] = layer[inside]

    def frame(self, moves=None, platform=None, looks=None, turn=0.0):
        """
        Paint every object at its place plus its move (None: the object is absent), then film it
        through `platform` (2x3, reference to this frame). Returns the frame and its labels.
        """
        moves = moves if moves is not None else [(0.0, 0.0)] * len(self.places)
        looks = looks if looks is not None else self.looks
        platform = np.eye(2, 3) if platform is None else np.asarray(platform, dtype=np.float64)
        world = self.ground.copy()
        quads = []
        length, width = (d * self.scale for d in self.CAR)
        for (x, y), angle, move, look in zip(self.places, self.angles, moves, looks, strict=True):
            if move is None:
                continue
            quad = rotated(x + move[0] * self.scale, y + move[1] * self.scale, length, width, angle + turn)
            self._paint(world, look, quad)
            quads.append(quad)
        image = cv2.warpAffine(world, platform, world.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
        filmed = [np.c_[q, np.ones(4)] @ platform.T for q in quads]
        return image, oriented(filmed) if filmed else empty_labels()

    def total_area(self):
        length, width = (d * self.scale for d in self.CAR)
        return len(self.places) * length * width / (self.shape[0] * self.shape[1])


def drift(dx=14.0, dy=-9.0, degrees=0.4, scale=1.0):
    """A platform motion: a small rotation about the frame center, then a translation."""
    center = (World.SIZE[1] * scale / 2, World.SIZE[0] * scale / 2)
    m = cv2.getRotationMatrix2D(center, degrees, 1.0)
    m[:, 2] += (dx * scale, dy * scale)
    return m


class TestPlatformMotion:
    def test_a_known_drift_is_recovered_at_the_corners(self):
        world = World()
        reference, ref_labels = world.frame()
        truth = drift()
        current, cur_labels = world.frame(platform=truth)
        estimate = platform_motion(reference, current, ref_labels, cur_labels)
        h, w = world.shape
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
        expected = corners @ np.vstack([truth, [0, 0, 1]]).T
        assert np.abs(corners @ estimate.T - expected).max() < 1.5

    def test_moving_objects_do_not_pull_the_estimate(self):
        # Every object moves the same way; the ground does not.
        world = World(count=40)
        reference, ref_labels = world.frame()
        current, cur_labels = world.frame(moves=[(40.0, 0.0)] * 40)
        estimate = platform_motion(reference, current, ref_labels, cur_labels)
        assert np.abs(estimate - np.eye(3)).max() < 0.5

    def test_frames_of_different_sizes_are_refused(self):
        with pytest.raises(ValueError):
            platform_motion(textured(), textured(320, 240))


class TestObjectPatches:
    def test_the_same_object_turned_half_a_circle_matches(self):
        world = World(count=1)
        reference, labels = world.frame()
        current, turned = world.frame(turn=180.0)
        a = object_patches(reference, labels.corners())
        b = object_patches(current, turned.corners())
        flipped = b.reshape(1, 14, 32)[:, ::-1, ::-1].reshape(1, -1)
        assert max((a @ b.T).item(), (a @ flipped.T).item()) > 0.9

    def test_different_objects_do_not(self):
        world = World(count=2)
        image, labels = world.frame()
        patches = object_patches(image, labels.corners())
        assert abs(float(patches[0] @ patches[1])) < 0.6

    def test_a_flat_patch_is_zero_rather_than_undefined(self):
        flat = np.full((100, 100, 3), 90, dtype=np.uint8)
        patch = object_patches(flat, rotated(50, 50, 30, 12, 0)[None])
        assert np.all(patch == 0)

    def test_box_dimensions_put_the_long_side_first(self):
        lengths, widths = box_dimensions(rotated(0, 0, 40, 10, 63)[None])
        assert lengths[0] == pytest.approx(40) and widths[0] == pytest.approx(10)


class TestAnchoredMotion:
    """Where the reference objects are in a later frame, on worlds whose answer is known."""

    @staticmethod
    def _fates(world, moves, **frame):
        reference, ref_labels = world.frame()
        current, cur_labels = world.frame(moves=moves, platform=frame.pop('platform', drift()), **frame)
        return anchored_motion(reference, current, ref_labels, cur_labels)

    def test_a_queue_is_static_whatever_the_platform_does(self):
        world = World()
        fates = self._fates(world, None)
        assert fates['static'] == pytest.approx(world.total_area(), rel=0.05)
        assert fates['displaced'] == 0.0 and fates['gone'] == pytest.approx(0.0, abs=1e-9)

    def test_one_stream_is_displaced_and_moves_together(self):
        world = World()
        fates = self._fates(world, [(70.0, 0.0)] * 30)
        assert fates['displaced'] == pytest.approx(world.total_area(), rel=0.05)
        assert fates['coherent'] == pytest.approx(fates['displaced'])
        assert fates['shift'][0] == pytest.approx(70.0, abs=3.0)

    def test_opposing_streams_are_not_one_flow(self):
        world = World()
        fates = self._fates(world, [(70.0, 0.0) if i % 2 else (-70.0, 0.0) for i in range(30)])
        assert fates['displaced'] == pytest.approx(world.total_area(), rel=0.05)
        assert fates['coherent'] / fates['displaced'] == pytest.approx(0.5, abs=0.04)

    def test_objects_that_left_are_gone(self):
        world = World()
        fates = self._fates(world, [None] * 15 + [(0.0, 0.0)] * 15)
        assert fates['gone'] == pytest.approx(world.total_area() / 2, rel=0.05)
        assert fates['static'] == pytest.approx(world.total_area() / 2, rel=0.05)

    def test_a_different_object_in_the_same_place_is_not_the_same_object(self):
        # A queue that emptied and refilled: occupied where it was, but by other objects.
        world = World()
        others = [world._look() for _ in range(30)]
        fates = self._fates(world, None, looks=others)
        assert fates['static'] < 0.1 * world.total_area()

    def test_two_look_alikes_do_not_both_claim_the_one_that_stayed(self):
        # Without mutual matching the second would read as displaced onto the first's place.
        world = World(count=2)
        world.looks[1] = world.looks[0].copy()
        reference, ref_labels = world.frame()
        current, cur_labels = world.frame(moves=[(0.0, 0.0), None])
        fates = anchored_motion(reference, current, ref_labels, cur_labels)
        assert fates['displaced'] == 0.0
        assert fates['gone'] == pytest.approx(world.total_area() / 2, rel=0.05)

    def test_an_object_of_another_size_in_the_same_place_is_not_the_same_object(self):
        world = World(count=1)
        reference, ref_labels = world.frame()
        (x, y), angle = world.places[0], world.angles[0]
        current = world.ground.copy()
        quad = rotated(x, y, World.CAR[0] * 1.6, World.CAR[1], angle)
        world._paint(current, cv2.resize(world.looks[0], (round(World.CAR[0] * 1.6), World.CAR[1])), quad)
        fates = anchored_motion(reference, current, ref_labels, oriented([quad]), motion=np.eye(3))
        assert fates['static'] == 0.0 and fates['gone'] > 0

    def test_a_small_move_is_not_a_displacement(self):
        # Under a tenth of an object width (24 px here): within box placement, not a move.
        world = World()
        fates = self._fates(world, [(1.0, 0.0)] * 30)
        assert fates['displaced'] == 0.0

    def test_a_stream_one_frame_apart_is_displaced_and_moves_together(self):
        # A quarter of an object width: what traffic covers between two consecutive frames.
        world = World()
        fates = self._fates(world, [(6.0, 0.0)] * 30)
        assert fates['displaced'] == pytest.approx(world.total_area(), rel=0.05)
        assert fates['coherent'] == pytest.approx(fates['displaced'])

    def test_opposing_streams_one_frame_apart_are_not_one_flow(self):
        world = World()
        fates = self._fates(world, [(6.0, 0.0) if i % 2 else (-6.0, 0.0) for i in range(30)])
        assert fates['displaced'] == pytest.approx(world.total_area(), rel=0.05)
        assert fates['coherent'] / fates['displaced'] == pytest.approx(0.5, abs=0.04)

    def test_a_stream_whose_members_drifted_apart_moves_together(self):
        # Speeds within a tenth of each other: a platoon, however far it went.
        world = World()
        fates = self._fates(world, [(66.0 + (6.0 if i % 2 else -6.0), 0.0) for i in range(30)])
        assert fates['coherent'] == pytest.approx(fates['displaced'])

    def test_streams_at_different_speeds_are_not_one_structure(self):
        # Same direction, one twice as fast as the other.
        world = World()
        fates = self._fates(world, [(36.0, 0.0) if i % 2 else (72.0, 0.0) for i in range(30)])
        assert fates['displaced'] == pytest.approx(world.total_area(), rel=0.05)
        assert fates['coherent'] / fates['displaced'] == pytest.approx(0.5, abs=0.04)

    def test_the_fates_do_not_depend_on_ground_resolution(self):
        small, large = World(scale=1.0), World(scale=2.0)
        moves = [(70.0, 0.0)] * 10 + [(0.0, 0.0)] * 10 + [None] * 10
        a = self._fates(small, moves, platform=drift(scale=1.0))
        b = self._fates(large, moves, platform=drift(scale=2.0))
        for fate in ('static', 'displaced', 'coherent', 'gone'):
            assert a[fate] == pytest.approx(b[fate], rel=0.05, abs=1e-6)

    def test_without_reference_objects_there_is_nothing(self):
        world = World()
        current, labels = world.frame()
        fates = anchored_motion(world.ground, current, empty_labels(), labels)
        assert all(fates[f] == 0.0 for f in ('static', 'displaced', 'coherent', 'gone'))

    def test_an_empty_later_frame_means_everything_is_gone(self):
        world = World()
        reference, labels = world.frame()
        fates = anchored_motion(reference, world.ground, labels, empty_labels())
        assert fates['gone'] == pytest.approx(world.total_area(), rel=0.05)


class TestSummarizeFates:
    def test_offsets_weigh_equally_and_coherence_is_a_share_of_the_displaced(self):
        per = [
            {'static': 0.02, 'displaced': 0.01, 'coherent': 0.01, 'gone': 0.0, 'n_displaced': 4},
            {'static': 0.00, 'displaced': 0.03, 'coherent': 0.01, 'gone': 0.01, 'n_displaced': 6},
        ]
        summary = summarize_fates(per)
        assert summary['static'] == pytest.approx(0.01)
        assert summary['displaced'] == pytest.approx(0.02)
        assert summary['coherence'] == pytest.approx(0.5)

    def test_a_few_displaced_objects_have_no_coherence(self):
        # One or two displaced objects always agree with themselves.
        per = [{'static': 0.02, 'displaced': 0.001, 'coherent': 0.001, 'gone': 0.0, 'n_displaced': 1}] * 3
        assert np.isnan(summarize_fates(per)['coherence'])

    def test_nothing_displaced_has_no_coherence(self):
        summary = summarize_fates([{'static': 0.02, 'displaced': 0.0, 'coherent': 0.0, 'gone': 0.0}])
        assert np.isnan(summary['coherence'])

    def test_no_offsets_is_undefined(self):
        assert np.isnan(summarize_fates([])['static'])
