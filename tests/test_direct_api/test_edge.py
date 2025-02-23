"""
build123d imports

name: test_edge.py
by:   Gumyr
date: January 22, 2025

desc:
    This python module contains tests for the build123d project.

license:

    Copyright 2025 Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import math
import unittest

from build123d.build_enums import AngularDirection, GeomType, Transition
from build123d.geometry import Axis, Plane, Vector
from build123d.objects_curve import CenterArc, EllipticalCenterArc
from build123d.objects_sketch import Circle, Rectangle, RegularPolygon
from build123d.operations_generic import sweep
from build123d.topology import Edge


class TestEdge(unittest.TestCase):
    def test_close(self):
        self.assertAlmostEqual(
            Edge.make_circle(1, end_angle=180).close().length, math.pi + 2, 5
        )
        self.assertAlmostEqual(Edge.make_circle(1).close().length, 2 * math.pi, 5)

    def test_make_half_circle(self):
        half_circle = Edge.make_circle(radius=1, start_angle=0, end_angle=180)
        self.assertAlmostEqual(half_circle.start_point(), (1, 0, 0), 3)
        self.assertAlmostEqual(half_circle.end_point(), (-1, 0, 0), 3)

    def test_make_half_circle2(self):
        half_circle = Edge.make_circle(radius=1, start_angle=270, end_angle=90)
        self.assertAlmostEqual(half_circle.start_point(), (0, -1, 0), 3)
        self.assertAlmostEqual(half_circle.end_point(), (0, 1, 0), 3)

    def test_make_clockwise_half_circle(self):
        half_circle = Edge.make_circle(
            radius=1,
            start_angle=180,
            end_angle=0,
            angular_direction=AngularDirection.CLOCKWISE,
        )
        self.assertAlmostEqual(half_circle.end_point(), (1, 0, 0), 3)
        self.assertAlmostEqual(half_circle.start_point(), (-1, 0, 0), 3)

    def test_make_clockwise_half_circle2(self):
        half_circle = Edge.make_circle(
            radius=1,
            start_angle=90,
            end_angle=-90,
            angular_direction=AngularDirection.CLOCKWISE,
        )
        self.assertAlmostEqual(half_circle.start_point(), (0, 1, 0), 3)
        self.assertAlmostEqual(half_circle.end_point(), (0, -1, 0), 3)

    def test_arc_center(self):
        self.assertAlmostEqual(Edge.make_ellipse(2, 1).arc_center, (0, 0, 0), 5)
        with self.assertRaises(ValueError):
            Edge.make_line((0, 0, 0), (0, 0, 1)).arc_center

    def test_spline_with_parameters(self):
        spline = Edge.make_spline(
            points=[(0, 0, 0), (1, 1, 0), (2, 0, 0)], parameters=[0.0, 0.4, 1.0]
        )
        self.assertAlmostEqual(spline.end_point(), (2, 0, 0), 5)
        with self.assertRaises(ValueError):
            Edge.make_spline(
                points=[(0, 0, 0), (1, 1, 0), (2, 0, 0)], parameters=[0.0, 1.0]
            )
        with self.assertRaises(ValueError):
            Edge.make_spline(
                points=[(0, 0, 0), (1, 1, 0), (2, 0, 0)], tangents=[(1, 1, 0)]
            )

    def test_spline_approx(self):
        spline = Edge.make_spline_approx([(0, 0), (1, 1), (2, 1), (3, 0)])
        self.assertAlmostEqual(spline.end_point(), (3, 0, 0), 5)
        spline = Edge.make_spline_approx(
            [(0, 0), (1, 1), (2, 1), (3, 0)], smoothing=(1.0, 5.0, 10.0)
        )
        self.assertAlmostEqual(spline.end_point(), (3, 0, 0), 5)

    def test_distribute_locations(self):
        line = Edge.make_line((0, 0, 0), (10, 0, 0))
        locs = line.distribute_locations(3)
        for i, x in enumerate([0, 5, 10]):
            self.assertAlmostEqual(locs[i].position, (x, 0, 0), 5)
        self.assertAlmostEqual(locs[0].orientation, (0, 90, 180), 5)

        locs = line.distribute_locations(3, positions_only=True)
        for i, x in enumerate([0, 5, 10]):
            self.assertAlmostEqual(locs[i].position, (x, 0, 0), 5)
        self.assertAlmostEqual(locs[0].orientation, (0, 0, 0), 5)

    def test_to_wire(self):
        edge = Edge.make_line((0, 0, 0), (1, 1, 1))
        for end in [0, 1]:
            self.assertAlmostEqual(
                edge.position_at(end),
                edge.to_wire().position_at(end),
                5,
            )

    def test_arc_center2(self):
        edges = [
            Edge.make_circle(1, plane=Plane((1, 2, 3)), end_angle=30),
            Edge.make_ellipse(1, 0.5, plane=Plane((1, 2, 3)), end_angle=30),
        ]
        for edge in edges:
            self.assertAlmostEqual(edge.arc_center, (1, 2, 3), 5)
        with self.assertRaises(ValueError):
            Edge.make_line((0, 0), (1, 1)).arc_center

    def test_find_intersection_points(self):
        circle = Edge.make_circle(1)
        line = Edge.make_line((0, -2), (0, 2))
        crosses = circle.find_intersection_points(line)
        for target, actual in zip([(0, 1, 0), (0, -1, 0)], crosses):
            self.assertAlmostEqual(actual, target, 5)

        with self.assertRaises(ValueError):
            circle.find_intersection_points(Edge.make_line((0, 0, -1), (0, 0, 1)))
        with self.assertRaises(ValueError):
            circle.find_intersection_points(Edge.make_line((0, 0, -1), (0, 0, 1)))

        self_intersect = Edge.make_spline([(-3, 2), (3, -2), (4, 0), (3, 2), (-3, -2)])
        self.assertAlmostEqual(
            self_intersect.find_intersection_points()[0],
            (-2.6861636507066047, 0, 0),
            5,
        )
        line = Edge.make_line((1, -2), (1, 2))
        crosses = line.find_intersection_points(Axis.X)
        self.assertAlmostEqual(crosses[0], (1, 0, 0), 5)

        with self.assertRaises(ValueError):
            line.find_intersection_points(Plane.YZ)

    # def test_intersections_tolerance(self):

    # Multiple operands not currently supported

    # r1 = ShapeList() + (PolarLocations(1, 4) * Edge.make_line((0, -1), (0, 1)))
    # l1 = Edge.make_line((1, 0), (2, 0))
    # i1 = l1.intersect(*r1)

    # r2 = Rectangle(2, 2).edges()
    # l2 = Pos(1) * Edge.make_line((0, 0), (1, 0))
    # i2 = l2.intersect(*r2)

    # self.assertEqual(len(i1.vertices()), len(i2.vertices()))

    def test_trim(self):
        line = Edge.make_line((-2, 0), (2, 0))
        self.assertAlmostEqual(line.trim(0.25, 0.75).position_at(0), (-1, 0, 0), 5)
        self.assertAlmostEqual(line.trim(0.25, 0.75).position_at(1), (1, 0, 0), 5)
        with self.assertRaises(ValueError):
            line.trim(0.75, 0.25)

    def test_trim_to_length(self):

        e1 = Edge.make_line((0, 0), (10, 10))
        e1_trim = e1.trim_to_length(0.0, 10)
        self.assertAlmostEqual(e1_trim.length, 10, 5)

        e2 = Edge.make_circle(10, start_angle=0, end_angle=90)
        e2_trim = e2.trim_to_length(0.5, 1)
        self.assertAlmostEqual(e2_trim.length, 1, 5)
        self.assertAlmostEqual(
            e2_trim.position_at(0), Vector(10, 0, 0).rotate(Axis.Z, 45), 5
        )

        e3 = Edge.make_spline(
            [(0, 10, 0), (-4, 5, 2), (0, 0, 0)], tangents=[(-1, 0), (1, 0)]
        )
        e3_trim = e3.trim_to_length(0, 7)
        self.assertAlmostEqual(e3_trim.length, 7, 5)

        a4 = Axis((0, 0, 0), (1, 1, 1))
        e4_trim = Edge(a4).trim_to_length(0.5, 2)
        self.assertAlmostEqual(e4_trim.length, 2, 5)

    def test_bezier(self):
        with self.assertRaises(ValueError):
            Edge.make_bezier((1, 1))
        cntl_pnts = [(1, 2, 3)] * 30
        with self.assertRaises(ValueError):
            Edge.make_bezier(*cntl_pnts)
        with self.assertRaises(ValueError):
            Edge.make_bezier((0, 0, 0), (1, 1, 1), weights=[1.0])

        bezier = Edge.make_bezier((0, 0), (0, 1), (1, 1), (1, 0))
        bbox = bezier.bounding_box()
        self.assertAlmostEqual(bbox.min, (0, 0, 0), 5)
        self.assertAlmostEqual(bbox.max, (1, 0.75, 0), 5)

    def test_mid_way(self):
        mid = Edge.make_mid_way(
            Edge.make_line((0, 0), (0, 1)), Edge.make_line((1, 0), (1, 1)), 0.25
        )
        self.assertAlmostEqual(mid.position_at(0), (0.25, 0, 0), 5)
        self.assertAlmostEqual(mid.position_at(1), (0.25, 1, 0), 5)

    def test_distribute_locations2(self):
        with self.assertRaises(ValueError):
            Edge.make_circle(1).distribute_locations(1)

        locs = Edge.make_circle(1).distribute_locations(5, positions_only=True)
        for i, loc in enumerate(locs):
            self.assertAlmostEqual(
                loc.position,
                Vector(1, 0, 0).rotate(Axis.Z, i * 90).to_tuple(),
                5,
            )
            self.assertAlmostEqual(loc.orientation, (0, 0, 0), 5)

    def test_find_tangent(self):
        circle = Edge.make_circle(1)
        parm = circle.find_tangent(135)[0]
        self.assertAlmostEqual(
            circle @ parm, (math.sqrt(2) / 2, math.sqrt(2) / 2, 0), 5
        )
        line = Edge.make_line((0, 0), (1, 1))
        parm = line.find_tangent(45)[0]
        self.assertAlmostEqual(parm, 0, 5)
        parm = line.find_tangent(0)
        self.assertEqual(len(parm), 0)

    def test_param_at_point(self):
        u = Edge.make_circle(1).param_at_point((0, 1))
        self.assertAlmostEqual(u, 0.25, 5)

        u = 0.3
        edge = Edge.make_line((0, 0), (34, 56))
        pnt = edge.position_at(u)
        self.assertAlmostEqual(edge.param_at_point(pnt), u, 5)

        ca = CenterArc((0, 0), 1, -200, 220).edge()
        for u in [0.3, 1.0]:
            pnt = ca.position_at(u)
            self.assertAlmostEqual(ca.param_at_point(pnt), u, 5)

        ea = EllipticalCenterArc((15, 0), 10, 5, start_angle=90, end_angle=270).edge()
        for u in [0.3, 0.9]:
            pnt = ea.position_at(u)
            self.assertAlmostEqual(ea.param_at_point(pnt), u, 5)

        with self.assertRaises(ValueError):
            edge.param_at_point((-1, 1))

    def test_conical_helix(self):
        helix = Edge.make_helix(1, 4, 1, normal=(-1, 0, 0), angle=10, lefthand=True)
        self.assertAlmostEqual(helix.bounding_box().min.X, -4, 5)

    def test_reverse(self):
        e1 = Edge.make_line((0, 0), (1, 1))
        self.assertAlmostEqual(e1 @ 0.1, (0.1, 0.1, 0), 5)
        self.assertAlmostEqual(e1.reversed() @ 0.1, (0.9, 0.9, 0), 5)

        e2 = Edge.make_circle(1, start_angle=0, end_angle=180)
        e2r = e2.reversed()
        self.assertAlmostEqual((e2 @ 0.1).X, -(e2r @ 0.1).X, 5)

    def test_init(self):
        with self.assertRaises(TypeError):
            Edge(direction=(1, 0, 0))

    def test_is_interior(self):
        path = RegularPolygon(5, 5).face().outer_wire()
        profile = path.location_at(0) * (Circle(0.6) & Rectangle(2, 1))
        target = sweep(profile, path, transition=Transition.RIGHT)
        inside_edges = target.edges().filter_by(lambda e: e.is_interior)
        self.assertEqual(len(inside_edges), 5)
        self.assertTrue(all(e.geom_type == GeomType.ELLIPSE for e in inside_edges))


if __name__ == "__main__":
    unittest.main()
