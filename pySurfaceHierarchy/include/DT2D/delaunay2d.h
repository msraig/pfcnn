#pragma once

//wrapper for 2D delaunay triangulation

#include <vector>
#include "delabella.h"

class Delaunay2D {
public:
	// Input: array of pts. Output: a Delaunay triangulation of the pts, as triangles indexing the pts.
	Delaunay2D(const std::vector<OpenMesh::Vec2d>& pts, std::vector<std::vector<int>>& tris) {
		IDelaBella* idb = IDelaBella::Create();
		int verts = idb->Triangulate(pts.size(), &pts[0][0], &pts[0][1], sizeof(OpenMesh::Vec2d));
		if (verts > 0)
		{
			int ntri = verts / 3; tris.resize(ntri, std::vector<int>(3, -1));
			const DelaBella_Triangle* dela = idb->GetFirstDelaunayTriangle();
			for (int i = 0; i < ntri; i++)
			{
				for (int j = 0; j < 3; j++)	tris[i][j] = dela->v[j]->i;
				dela = dela->next;
			}
		}
		else
		{
			printf("DT2D: all points coplanar. Exiting...\n"); std::exit(0);
		}
		idb->Destroy();
	}
	Delaunay2D()
	{
		return;
	};
	bool Triangulate(const std::vector<OpenMesh::Vec2d>& pts, std::vector<std::vector<int>>& tris) {
		IDelaBella* idb = IDelaBella::Create();
		int verts = idb->Triangulate(pts.size(), &pts[0][0], &pts[0][1], sizeof(OpenMesh::Vec2d));
		if (verts>0)
		{
			int ntri = verts / 3; tris.resize(ntri, std::vector<int>(3,-1));
			const DelaBella_Triangle* dela = idb->GetFirstDelaunayTriangle();
			for (int i = 0; i<ntri; i++)
			{
				for (int j = 0; j < 3; j++)	tris[i][j] = dela->v[j]->i;
				dela = dela->next;
			}
		}
		else
		{
			printf("DT2D: all points coplanar. Exiting...\n");// std::exit(0);
			/*for (int i = 0; i < pts.size(); i++)
			{
				std::cout << pts[i] << std::endl;
			}
			system("pause");*/
			idb->Destroy();
			return false;
		}
		idb->Destroy();
		return true;
	}
};