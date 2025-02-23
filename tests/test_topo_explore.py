from typing import Optional
import unittest

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.gp import gp_Pnt, gp_Pln
from OCP.TopoDS import TopoDS_Face, TopoDS_Shape
from build123d.build_enums import SortBy

from build123d.objects_part import Box
from build123d.geometry import (
    Axis,
    Vector,
    VectorLike,
)
from build123d.topology import (
    Edge,
    Face,
    Shell,
    Wire,
    offset_topods_face,
    topo_explore_connected_edges,
    topo_explore_connected_faces,
    topo_explore_common_vertex,
)


class DirectApiTestCase(unittest.TestCase):
    def assertTupleAlmostEquals(
        self,
        first: tuple[float, ...],
        second: tuple[float, ...],
        places: int,
        msg: Optional[str] = None,
    ):
        """Check Tuples"""
        self.assertEqual(len(second), len(first))
        for i, j in zip(second, first):
            self.assertAlmostEqual(i, j, places, msg=msg)

    def assertVectorAlmostEquals(
        self, first: Vector, second: VectorLike, places: int, msg: Optional[str] = None
    ):
        second_vector = Vector(second)
        self.assertAlmostEqual(first.X, second_vector.X, places, msg=msg)
        self.assertAlmostEqual(first.Y, second_vector.Y, places, msg=msg)
        self.assertAlmostEqual(first.Z, second_vector.Z, places, msg=msg)


class TestTopoExplore(DirectApiTestCase):

    def test_topo_explore_connected_edges(self):
        # 2D
        triangle = Face(
            Wire(
                [
                    Edge.make_line((0, 0), (2.5, 0)),
                    Edge.make_line((2.5, 0), (4.0)),
                    Edge.make_line((4, 0), (0, 3)),
                    Edge.make_line((0, 3), (0, 1.25)),
                    Edge.make_line((0, 1.25), (0, 0)),
                ]
            )
        )

        hypotenuse = triangle.edges().sort_by(SortBy.LENGTH)[-1]
        connected_edges = topo_explore_connected_edges(hypotenuse).sort_by(
            SortBy.LENGTH
        )
        self.assertAlmostEqual(connected_edges[0].length, 1.5, 5)
        self.assertAlmostEqual(connected_edges[1].length, 1.75, 5)

        # 3D
        box = Box(1, 1, 1)
        first_edge = box.edges()[0]
        connected_edges = topo_explore_connected_edges(first_edge)
        self.assertEqual(len(connected_edges), 4)

        self.assertEqual(len(topo_explore_connected_edges(hypotenuse, parent=box)), 0)

        # 2 Edges
        l1 = Edge.make_spline([(-1, 0), (1, 0)], tangents=((0, -8), (0, 8)), scale=True)
        l2 = Edge.make_line(l1 @ 0, l1 @ 1)
        face = Face(Wire([l1, l2]))
        connected_edges = topo_explore_connected_edges(face.edges()[0])
        self.assertEqual(len(connected_edges), 1)

    def test_topo_explore_connected_edges_errors(self):
        # No parent case
        with self.assertRaises(ValueError):
            topo_explore_connected_edges(Edge())

        # Null edge case
        null_edge = Wire.make_rect(1, 1).edges()[0]
        null_edge.wrapped = None
        with self.assertRaises(ValueError):
            topo_explore_connected_edges(null_edge)

    def test_topo_explore_common_vertex(self):
        triangle = Face(
            Wire(
                [
                    Edge.make_line((0, 0), (4, 0)),
                    Edge.make_line((4, 0), (0, 3)),
                    Edge.make_line((0, 3), (0, 0)),
                ]
            )
        )
        hypotenuse = triangle.edges().sort_by(SortBy.LENGTH)[-1]
        base = triangle.edges().sort_by(Axis.Y)[0]
        common_vertex = topo_explore_common_vertex(hypotenuse, base)
        self.assertIsNotNone(common_vertex)
        self.assertVectorAlmostEquals(Vector(common_vertex), (4, 0, 0), 5)
        self.assertIsNone(
            topo_explore_common_vertex(hypotenuse, Edge.make_line((0, 0), (4, 0)))
        )


class TestOffsetTopodsFace(unittest.TestCase):
    def setUp(self):
        # Create a simple planar face for testing
        self.face = Face.make_rect(1, 1).wrapped

    def get_face_center(self, face: TopoDS_Face) -> tuple:
        """Calculate the center of a face"""
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        center = props.CentreOfMass()
        return (center.X(), center.Y(), center.Z())

    def test_offset_topods_face(self):
        # Offset the face by a positive amount
        offset_amount = 1.0
        original_center = self.get_face_center(self.face)
        offset_shape = offset_topods_face(self.face, offset_amount)
        offset_center = self.get_face_center(offset_shape)
        self.assertIsInstance(offset_shape, TopoDS_Shape)
        self.assertAlmostEqual(Vector(0, 0, 1), offset_center)

        # Offset the face by a negative amount
        offset_amount = -1.0
        offset_shape = offset_topods_face(self.face, offset_amount)
        offset_center = self.get_face_center(offset_shape)
        self.assertIsInstance(offset_shape, TopoDS_Shape)
        self.assertAlmostEqual(Vector(0, 0, -1), offset_center)

    def test_offset_topods_face_zero(self):
        # Offset the face by zero amount
        offset_amount = 0.0
        original_center = self.get_face_center(self.face)
        offset_shape = offset_topods_face(self.face, offset_amount)
        offset_center = self.get_face_center(offset_shape)
        self.assertIsInstance(offset_shape, TopoDS_Shape)
        self.assertAlmostEqual(Vector(original_center), offset_center)


class TestTopoExploreConnectedFaces(unittest.TestCase):
    def setUp(self):
        # Create a shell with 4 faces
        walls = Shell.extrude(Wire.make_rect(1, 1), (0, 0, 1))
        diagonal = Axis((0, 0, 0), (1, 1, 0))

        # Extract the edge that is connected to two faces
        self.connected_edge = walls.edges().filter_by(Axis.Z).sort_by(diagonal)[-1]

        # Create an edge that is only connected to one face
        self.unconnected_edge = Face.make_rect(1, 1).edges()[0]

    def test_topo_explore_connected_faces(self):
        # Add the edge to the faces
        faces = topo_explore_connected_faces(self.connected_edge)
        self.assertEqual(len(faces), 2)

    def test_topo_explore_connected_faces_invalid(self):
        # Test with an edge that is not connected to two faces
        with self.assertRaises(RuntimeError):
            topo_explore_connected_faces(self.unconnected_edge)

        # No parent case
        with self.assertRaises(ValueError):
            topo_explore_connected_faces(Edge())


if __name__ == "__main__":
    unittest.main()
