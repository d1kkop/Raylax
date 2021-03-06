/*
Copyright 2017 Ioannis Makris

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// This file was generated by glatter.py script.



#ifdef GLATTER_GLU
#if defined(__glu_h__)
#if defined(__GLU_H__)
#define gluBeginCurve(nobj) glatter_gluBeginCurve((nobj))
GLATTER_UBLOCK(void, APIENTRY, gluBeginCurve, (GLUnurbs *nobj))
#define gluBeginPolygon(tess) glatter_gluBeginPolygon((tess))
GLATTER_UBLOCK(void, APIENTRY, gluBeginPolygon, (GLUtesselator *tess))
#define gluBeginSurface(nobj) glatter_gluBeginSurface((nobj))
GLATTER_UBLOCK(void, APIENTRY, gluBeginSurface, (GLUnurbs *nobj))
#define gluBeginTrim(nobj) glatter_gluBeginTrim((nobj))
GLATTER_UBLOCK(void, APIENTRY, gluBeginTrim, (GLUnurbs *nobj))
#define gluBuild1DMipmaps(target, components, width, format, type, data) glatter_gluBuild1DMipmaps((target), (components), (width), (format), (type), (data))
GLATTER_UBLOCK(int, APIENTRY, gluBuild1DMipmaps, (GLenum target, GLint components, GLint width, GLenum format, GLenum type, const void *data))
#define gluBuild2DMipmaps(target, components, width, height, format, type, data) glatter_gluBuild2DMipmaps((target), (components), (width), (height), (format), (type), (data))
GLATTER_UBLOCK(int, APIENTRY, gluBuild2DMipmaps, (GLenum target, GLint components, GLint width, GLint height, GLenum format, GLenum type, const void *data))
#define gluCylinder(qobj, baseRadius, topRadius, height, slices, stacks) glatter_gluCylinder((qobj), (baseRadius), (topRadius), (height), (slices), (stacks))
GLATTER_UBLOCK(void, APIENTRY, gluCylinder, (GLUquadric *qobj, GLdouble baseRadius, GLdouble topRadius, GLdouble height, GLint slices, GLint stacks))
#define gluDeleteNurbsRenderer(nobj) glatter_gluDeleteNurbsRenderer((nobj))
GLATTER_UBLOCK(void, APIENTRY, gluDeleteNurbsRenderer, (GLUnurbs *nobj))
#define gluDeleteQuadric(state) glatter_gluDeleteQuadric((state))
GLATTER_UBLOCK(void, APIENTRY, gluDeleteQuadric, (GLUquadric *state))
#define gluDeleteTess(tess) glatter_gluDeleteTess((tess))
GLATTER_UBLOCK(void, APIENTRY, gluDeleteTess, (GLUtesselator *tess))
#define gluDisk(qobj, innerRadius, outerRadius, slices, loops) glatter_gluDisk((qobj), (innerRadius), (outerRadius), (slices), (loops))
GLATTER_UBLOCK(void, APIENTRY, gluDisk, (GLUquadric *qobj, GLdouble innerRadius, GLdouble outerRadius, GLint slices, GLint loops))
#define gluEndCurve(nobj) glatter_gluEndCurve((nobj))
GLATTER_UBLOCK(void, APIENTRY, gluEndCurve, (GLUnurbs *nobj))
#define gluEndPolygon(tess) glatter_gluEndPolygon((tess))
GLATTER_UBLOCK(void, APIENTRY, gluEndPolygon, (GLUtesselator *tess))
#define gluEndSurface(nobj) glatter_gluEndSurface((nobj))
GLATTER_UBLOCK(void, APIENTRY, gluEndSurface, (GLUnurbs *nobj))
#define gluEndTrim(nobj) glatter_gluEndTrim((nobj))
GLATTER_UBLOCK(void, APIENTRY, gluEndTrim, (GLUnurbs *nobj))
#define gluErrorString(errCode) glatter_gluErrorString((errCode))
GLATTER_UBLOCK(const GLubyte*, APIENTRY, gluErrorString, (GLenum errCode))
#define gluErrorUnicodeStringEXT(errCode) glatter_gluErrorUnicodeStringEXT((errCode))
GLATTER_UBLOCK(const wchar_t*, APIENTRY, gluErrorUnicodeStringEXT, (GLenum errCode))
#define gluGetString(name) glatter_gluGetString((name))
GLATTER_UBLOCK(const GLubyte*, APIENTRY, gluGetString, (GLenum name))
#define gluGetTessProperty(tess, which, value) glatter_gluGetTessProperty((tess), (which), (value))
GLATTER_UBLOCK(void, APIENTRY, gluGetTessProperty, (GLUtesselator *tess, GLenum which, GLdouble *value))
#define gluLookAt(eyex, eyey, eyez, centerx, centery, centerz, upx, upy, upz) glatter_gluLookAt((eyex), (eyey), (eyez), (centerx), (centery), (centerz), (upx), (upy), (upz))
GLATTER_UBLOCK(void, APIENTRY, gluLookAt, (GLdouble eyex, GLdouble eyey, GLdouble eyez, GLdouble centerx, GLdouble centery, GLdouble centerz, GLdouble upx, GLdouble upy, GLdouble upz))
#define gluNewNurbsRenderer() glatter_gluNewNurbsRenderer()
GLATTER_UBLOCK(GLUnurbs*, APIENTRY, gluNewNurbsRenderer, (void))
#define gluNewQuadric() glatter_gluNewQuadric()
GLATTER_UBLOCK(GLUquadric*, APIENTRY, gluNewQuadric, (void))
#define gluNewTess() glatter_gluNewTess()
GLATTER_UBLOCK(GLUtesselator*, APIENTRY, gluNewTess, (void))
#define gluNextContour(tess, type) glatter_gluNextContour((tess), (type))
GLATTER_UBLOCK(void, APIENTRY, gluNextContour, (GLUtesselator *tess, GLenum type))
#define gluNurbsCurve(nobj, nknots, knot, stride, ctlarray, order, type) glatter_gluNurbsCurve((nobj), (nknots), (knot), (stride), (ctlarray), (order), (type))
GLATTER_UBLOCK(void, APIENTRY, gluNurbsCurve, (GLUnurbs *nobj, GLint nknots, GLfloat *knot, GLint stride, GLfloat *ctlarray, GLint order, GLenum type))
#define gluOrtho2D(left, right, bottom, top) glatter_gluOrtho2D((left), (right), (bottom), (top))
GLATTER_UBLOCK(void, APIENTRY, gluOrtho2D, (GLdouble left, GLdouble right, GLdouble bottom, GLdouble top))
#define gluPartialDisk(qobj, innerRadius, outerRadius, slices, loops, startAngle, sweepAngle) glatter_gluPartialDisk((qobj), (innerRadius), (outerRadius), (slices), (loops), (startAngle), (sweepAngle))
GLATTER_UBLOCK(void, APIENTRY, gluPartialDisk, (GLUquadric *qobj, GLdouble innerRadius, GLdouble outerRadius, GLint slices, GLint loops, GLdouble startAngle, GLdouble sweepAngle))
#define gluPerspective(fovy, aspect, zNear, zFar) glatter_gluPerspective((fovy), (aspect), (zNear), (zFar))
GLATTER_UBLOCK(void, APIENTRY, gluPerspective, (GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar))
#define gluPickMatrix(x, y, width, height, viewport) glatter_gluPickMatrix((x), (y), (width), (height), (viewport))
GLATTER_UBLOCK(void, APIENTRY, gluPickMatrix, (GLdouble x, GLdouble y, GLdouble width, GLdouble height, GLint viewport[4]))
#define gluProject(objx, objy, objz, modelMatrix, projMatrix, viewport, winx, winy, winz) glatter_gluProject((objx), (objy), (objz), (modelMatrix), (projMatrix), (viewport), (winx), (winy), (winz))
GLATTER_UBLOCK(int, APIENTRY, gluProject, (GLdouble objx, GLdouble objy, GLdouble objz, const GLdouble modelMatrix[16], const GLdouble projMatrix[16], const GLint viewport[4], GLdouble *winx, GLdouble *winy, GLdouble *winz))
#define gluPwlCurve(nobj, count, array, stride, type) glatter_gluPwlCurve((nobj), (count), (array), (stride), (type))
GLATTER_UBLOCK(void, APIENTRY, gluPwlCurve, (GLUnurbs *nobj, GLint count, GLfloat *array, GLint stride, GLenum type))
#define gluQuadricCallback(qobj, which, fn) glatter_gluQuadricCallback((qobj), (which), (fn))
GLATTER_UBLOCK(void, APIENTRY, gluQuadricCallback, (GLUquadric *qobj, GLenum which, void (CALLBACK* fn)()))
#define gluQuadricDrawStyle(quadObject, drawStyle) glatter_gluQuadricDrawStyle((quadObject), (drawStyle))
GLATTER_UBLOCK(void, APIENTRY, gluQuadricDrawStyle, (GLUquadric *quadObject, GLenum drawStyle))
#define gluQuadricNormals(quadObject, normals) glatter_gluQuadricNormals((quadObject), (normals))
GLATTER_UBLOCK(void, APIENTRY, gluQuadricNormals, (GLUquadric *quadObject, GLenum normals))
#define gluQuadricOrientation(quadObject, orientation) glatter_gluQuadricOrientation((quadObject), (orientation))
GLATTER_UBLOCK(void, APIENTRY, gluQuadricOrientation, (GLUquadric *quadObject, GLenum orientation))
#define gluQuadricTexture(quadObject, textureCoords) glatter_gluQuadricTexture((quadObject), (textureCoords))
GLATTER_UBLOCK(void, APIENTRY, gluQuadricTexture, (GLUquadric *quadObject, GLboolean textureCoords))
#define gluScaleImage(format, widthin, heightin, typein, datain, widthout, heightout, typeout, dataout) glatter_gluScaleImage((format), (widthin), (heightin), (typein), (datain), (widthout), (heightout), (typeout), (dataout))
GLATTER_UBLOCK(int, APIENTRY, gluScaleImage, (GLenum format, GLint widthin, GLint heightin, GLenum typein, const void *datain, GLint widthout, GLint heightout, GLenum typeout, void *dataout))
#define gluSphere(qobj, radius, slices, stacks) glatter_gluSphere((qobj), (radius), (slices), (stacks))
GLATTER_UBLOCK(void, APIENTRY, gluSphere, (GLUquadric *qobj, GLdouble radius, GLint slices, GLint stacks))
#define gluTessBeginContour(tess) glatter_gluTessBeginContour((tess))
GLATTER_UBLOCK(void, APIENTRY, gluTessBeginContour, (GLUtesselator *tess))
#define gluTessBeginPolygon(tess, polygon_data) glatter_gluTessBeginPolygon((tess), (polygon_data))
GLATTER_UBLOCK(void, APIENTRY, gluTessBeginPolygon, (GLUtesselator *tess, void *polygon_data))
#define gluTessCallback(tess, which, fn) glatter_gluTessCallback((tess), (which), (fn))
GLATTER_UBLOCK(void, APIENTRY, gluTessCallback, (GLUtesselator *tess, GLenum which, void (CALLBACK *fn)()))
#define gluTessEndContour(tess) glatter_gluTessEndContour((tess))
GLATTER_UBLOCK(void, APIENTRY, gluTessEndContour, (GLUtesselator *tess))
#define gluTessEndPolygon(tess) glatter_gluTessEndPolygon((tess))
GLATTER_UBLOCK(void, APIENTRY, gluTessEndPolygon, (GLUtesselator *tess))
#define gluTessNormal(tess, x, y, z) glatter_gluTessNormal((tess), (x), (y), (z))
GLATTER_UBLOCK(void, APIENTRY, gluTessNormal, (GLUtesselator *tess, GLdouble x, GLdouble y, GLdouble z))
#define gluTessProperty(tess, which, value) glatter_gluTessProperty((tess), (which), (value))
GLATTER_UBLOCK(void, APIENTRY, gluTessProperty, (GLUtesselator *tess, GLenum which, GLdouble value))
#define gluTessVertex(tess, coords, data) glatter_gluTessVertex((tess), (coords), (data))
GLATTER_UBLOCK(void, APIENTRY, gluTessVertex, (GLUtesselator *tess, GLdouble coords[3], void *data))
#define gluUnProject(winx, winy, winz, modelMatrix, projMatrix, viewport, objx, objy, objz) glatter_gluUnProject((winx), (winy), (winz), (modelMatrix), (projMatrix), (viewport), (objx), (objy), (objz))
GLATTER_UBLOCK(int, APIENTRY, gluUnProject, (GLdouble winx, GLdouble winy, GLdouble winz, const GLdouble modelMatrix[16], const GLdouble projMatrix[16], const GLint viewport[4], GLdouble *objx, GLdouble *objy, GLdouble *objz))
#endif // defined(__GLU_H__)
#if !defined(__GLU_H__)
#define gluBeginCurve(nurb) glatter_gluBeginCurve((nurb))
GLATTER_UBLOCK(void, GLAPIENTRY, gluBeginCurve, (GLUnurbs* nurb))
#define gluBeginPolygon(tess) glatter_gluBeginPolygon((tess))
GLATTER_UBLOCK(void, GLAPIENTRY, gluBeginPolygon, (GLUtesselator* tess))
#define gluBeginSurface(nurb) glatter_gluBeginSurface((nurb))
GLATTER_UBLOCK(void, GLAPIENTRY, gluBeginSurface, (GLUnurbs* nurb))
#define gluBeginTrim(nurb) glatter_gluBeginTrim((nurb))
GLATTER_UBLOCK(void, GLAPIENTRY, gluBeginTrim, (GLUnurbs* nurb))
#endif // !defined(__GLU_H__)
#define gluBuild1DMipmapLevels(target, internalFormat, width, format, type, level, base, max, data) glatter_gluBuild1DMipmapLevels((target), (internalFormat), (width), (format), (type), (level), (base), (max), (data))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluBuild1DMipmapLevels, (GLenum target, GLint internalFormat, GLsizei width, GLenum format, GLenum type, GLint level, GLint base, GLint max, const void *data))
#if !defined(__GLU_H__)
#define gluBuild1DMipmaps(target, internalFormat, width, format, type, data) glatter_gluBuild1DMipmaps((target), (internalFormat), (width), (format), (type), (data))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluBuild1DMipmaps, (GLenum target, GLint internalFormat, GLsizei width, GLenum format, GLenum type, const void *data))
#endif // !defined(__GLU_H__)
#define gluBuild2DMipmapLevels(target, internalFormat, width, height, format, type, level, base, max, data) glatter_gluBuild2DMipmapLevels((target), (internalFormat), (width), (height), (format), (type), (level), (base), (max), (data))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluBuild2DMipmapLevels, (GLenum target, GLint internalFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, GLint level, GLint base, GLint max, const void *data))
#if !defined(__GLU_H__)
#define gluBuild2DMipmaps(target, internalFormat, width, height, format, type, data) glatter_gluBuild2DMipmaps((target), (internalFormat), (width), (height), (format), (type), (data))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluBuild2DMipmaps, (GLenum target, GLint internalFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *data))
#endif // !defined(__GLU_H__)
#define gluBuild3DMipmapLevels(target, internalFormat, width, height, depth, format, type, level, base, max, data) glatter_gluBuild3DMipmapLevels((target), (internalFormat), (width), (height), (depth), (format), (type), (level), (base), (max), (data))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluBuild3DMipmapLevels, (GLenum target, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLint level, GLint base, GLint max, const void *data))
#define gluBuild3DMipmaps(target, internalFormat, width, height, depth, format, type, data) glatter_gluBuild3DMipmaps((target), (internalFormat), (width), (height), (depth), (format), (type), (data))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluBuild3DMipmaps, (GLenum target, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data))
#define gluCheckExtension(extName, extString) glatter_gluCheckExtension((extName), (extString))
GLATTER_UBLOCK(GLboolean, GLAPIENTRY, gluCheckExtension, (const GLubyte *extName, const GLubyte *extString))
#if !defined(__GLU_H__)
#define gluCylinder(quad, base, top, height, slices, stacks) glatter_gluCylinder((quad), (base), (top), (height), (slices), (stacks))
GLATTER_UBLOCK(void, GLAPIENTRY, gluCylinder, (GLUquadric* quad, GLdouble base, GLdouble top, GLdouble height, GLint slices, GLint stacks))
#define gluDeleteNurbsRenderer(nurb) glatter_gluDeleteNurbsRenderer((nurb))
GLATTER_UBLOCK(void, GLAPIENTRY, gluDeleteNurbsRenderer, (GLUnurbs* nurb))
#define gluDeleteQuadric(quad) glatter_gluDeleteQuadric((quad))
GLATTER_UBLOCK(void, GLAPIENTRY, gluDeleteQuadric, (GLUquadric* quad))
#define gluDeleteTess(tess) glatter_gluDeleteTess((tess))
GLATTER_UBLOCK(void, GLAPIENTRY, gluDeleteTess, (GLUtesselator* tess))
#define gluDisk(quad, inner, outer, slices, loops) glatter_gluDisk((quad), (inner), (outer), (slices), (loops))
GLATTER_UBLOCK(void, GLAPIENTRY, gluDisk, (GLUquadric* quad, GLdouble inner, GLdouble outer, GLint slices, GLint loops))
#define gluEndCurve(nurb) glatter_gluEndCurve((nurb))
GLATTER_UBLOCK(void, GLAPIENTRY, gluEndCurve, (GLUnurbs* nurb))
#define gluEndPolygon(tess) glatter_gluEndPolygon((tess))
GLATTER_UBLOCK(void, GLAPIENTRY, gluEndPolygon, (GLUtesselator* tess))
#define gluEndSurface(nurb) glatter_gluEndSurface((nurb))
GLATTER_UBLOCK(void, GLAPIENTRY, gluEndSurface, (GLUnurbs* nurb))
#define gluEndTrim(nurb) glatter_gluEndTrim((nurb))
GLATTER_UBLOCK(void, GLAPIENTRY, gluEndTrim, (GLUnurbs* nurb))
#define gluErrorString(error) glatter_gluErrorString((error))
GLATTER_UBLOCK(const GLubyte *, GLAPIENTRY, gluErrorString, (GLenum error))
#endif // !defined(__GLU_H__)
#define gluGetNurbsProperty(nurb, property, data) glatter_gluGetNurbsProperty((nurb), (property), (data))
GLATTER_UBLOCK(void, GLAPIENTRY, gluGetNurbsProperty, (GLUnurbs* nurb, GLenum property, GLfloat* data))
#if !defined(__GLU_H__)
#define gluGetString(name) glatter_gluGetString((name))
GLATTER_UBLOCK(const GLubyte *, GLAPIENTRY, gluGetString, (GLenum name))
#define gluGetTessProperty(tess, which, data) glatter_gluGetTessProperty((tess), (which), (data))
GLATTER_UBLOCK(void, GLAPIENTRY, gluGetTessProperty, (GLUtesselator* tess, GLenum which, GLdouble* data))
#endif // !defined(__GLU_H__)
#define gluLoadSamplingMatrices(nurb, model, perspective, view) glatter_gluLoadSamplingMatrices((nurb), (model), (perspective), (view))
GLATTER_UBLOCK(void, GLAPIENTRY, gluLoadSamplingMatrices, (GLUnurbs* nurb, const GLfloat *model, const GLfloat *perspective, const GLint *view))
#if !defined(__GLU_H__)
#define gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ) glatter_gluLookAt((eyeX), (eyeY), (eyeZ), (centerX), (centerY), (centerZ), (upX), (upY), (upZ))
GLATTER_UBLOCK(void, GLAPIENTRY, gluLookAt, (GLdouble eyeX, GLdouble eyeY, GLdouble eyeZ, GLdouble centerX, GLdouble centerY, GLdouble centerZ, GLdouble upX, GLdouble upY, GLdouble upZ))
#define gluNewNurbsRenderer() glatter_gluNewNurbsRenderer()
GLATTER_UBLOCK(GLUnurbs*, GLAPIENTRY, gluNewNurbsRenderer, (void))
#define gluNewQuadric() glatter_gluNewQuadric()
GLATTER_UBLOCK(GLUquadric*, GLAPIENTRY, gluNewQuadric, (void))
#define gluNewTess() glatter_gluNewTess()
GLATTER_UBLOCK(GLUtesselator*, GLAPIENTRY, gluNewTess, (void))
#define gluNextContour(tess, type) glatter_gluNextContour((tess), (type))
GLATTER_UBLOCK(void, GLAPIENTRY, gluNextContour, (GLUtesselator* tess, GLenum type))
#endif // !defined(__GLU_H__)
#define gluNurbsCallback(nurb, which, CallBackFunc) glatter_gluNurbsCallback((nurb), (which), (CallBackFunc))
GLATTER_UBLOCK(void, GLAPIENTRY, gluNurbsCallback, (GLUnurbs* nurb, GLenum which, _GLUfuncptr CallBackFunc))
#define gluNurbsCallbackData(nurb, userData) glatter_gluNurbsCallbackData((nurb), (userData))
GLATTER_UBLOCK(void, GLAPIENTRY, gluNurbsCallbackData, (GLUnurbs* nurb, GLvoid* userData))
#define gluNurbsCallbackDataEXT(nurb, userData) glatter_gluNurbsCallbackDataEXT((nurb), (userData))
GLATTER_UBLOCK(void, GLAPIENTRY, gluNurbsCallbackDataEXT, (GLUnurbs* nurb, GLvoid* userData))
#if !defined(__GLU_H__)
#define gluNurbsCurve(nurb, knotCount, knots, stride, control, order, type) glatter_gluNurbsCurve((nurb), (knotCount), (knots), (stride), (control), (order), (type))
GLATTER_UBLOCK(void, GLAPIENTRY, gluNurbsCurve, (GLUnurbs* nurb, GLint knotCount, GLfloat *knots, GLint stride, GLfloat *control, GLint order, GLenum type))
#endif // !defined(__GLU_H__)
#define gluNurbsProperty(nurb, property, value) glatter_gluNurbsProperty((nurb), (property), (value))
GLATTER_UBLOCK(void, GLAPIENTRY, gluNurbsProperty, (GLUnurbs* nurb, GLenum property, GLfloat value))
#define gluNurbsSurface(nurb, sKnotCount, sKnots, tKnotCount, tKnots, sStride, tStride, control, sOrder, tOrder, type) glatter_gluNurbsSurface((nurb), (sKnotCount), (sKnots), (tKnotCount), (tKnots), (sStride), (tStride), (control), (sOrder), (tOrder), (type))
GLATTER_UBLOCK(void, GLAPIENTRY, gluNurbsSurface, (GLUnurbs* nurb, GLint sKnotCount, GLfloat* sKnots, GLint tKnotCount, GLfloat* tKnots, GLint sStride, GLint tStride, GLfloat* control, GLint sOrder, GLint tOrder, GLenum type))
#if !defined(__GLU_H__)
#define gluOrtho2D(left, right, bottom, top) glatter_gluOrtho2D((left), (right), (bottom), (top))
GLATTER_UBLOCK(void, GLAPIENTRY, gluOrtho2D, (GLdouble left, GLdouble right, GLdouble bottom, GLdouble top))
#define gluPartialDisk(quad, inner, outer, slices, loops, start, sweep) glatter_gluPartialDisk((quad), (inner), (outer), (slices), (loops), (start), (sweep))
GLATTER_UBLOCK(void, GLAPIENTRY, gluPartialDisk, (GLUquadric* quad, GLdouble inner, GLdouble outer, GLint slices, GLint loops, GLdouble start, GLdouble sweep))
#define gluPerspective(fovy, aspect, zNear, zFar) glatter_gluPerspective((fovy), (aspect), (zNear), (zFar))
GLATTER_UBLOCK(void, GLAPIENTRY, gluPerspective, (GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar))
#define gluPickMatrix(x, y, delX, delY, viewport) glatter_gluPickMatrix((x), (y), (delX), (delY), (viewport))
GLATTER_UBLOCK(void, GLAPIENTRY, gluPickMatrix, (GLdouble x, GLdouble y, GLdouble delX, GLdouble delY, GLint *viewport))
#define gluProject(objX, objY, objZ, model, proj, view, winX, winY, winZ) glatter_gluProject((objX), (objY), (objZ), (model), (proj), (view), (winX), (winY), (winZ))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluProject, (GLdouble objX, GLdouble objY, GLdouble objZ, const GLdouble *model, const GLdouble *proj, const GLint *view, GLdouble* winX, GLdouble* winY, GLdouble* winZ))
#define gluPwlCurve(nurb, count, data, stride, type) glatter_gluPwlCurve((nurb), (count), (data), (stride), (type))
GLATTER_UBLOCK(void, GLAPIENTRY, gluPwlCurve, (GLUnurbs* nurb, GLint count, GLfloat* data, GLint stride, GLenum type))
#define gluQuadricCallback(quad, which, CallBackFunc) glatter_gluQuadricCallback((quad), (which), (CallBackFunc))
GLATTER_UBLOCK(void, GLAPIENTRY, gluQuadricCallback, (GLUquadric* quad, GLenum which, _GLUfuncptr CallBackFunc))
#define gluQuadricDrawStyle(quad, draw) glatter_gluQuadricDrawStyle((quad), (draw))
GLATTER_UBLOCK(void, GLAPIENTRY, gluQuadricDrawStyle, (GLUquadric* quad, GLenum draw))
#define gluQuadricNormals(quad, normal) glatter_gluQuadricNormals((quad), (normal))
GLATTER_UBLOCK(void, GLAPIENTRY, gluQuadricNormals, (GLUquadric* quad, GLenum normal))
#define gluQuadricOrientation(quad, orientation) glatter_gluQuadricOrientation((quad), (orientation))
GLATTER_UBLOCK(void, GLAPIENTRY, gluQuadricOrientation, (GLUquadric* quad, GLenum orientation))
#define gluQuadricTexture(quad, texture) glatter_gluQuadricTexture((quad), (texture))
GLATTER_UBLOCK(void, GLAPIENTRY, gluQuadricTexture, (GLUquadric* quad, GLboolean texture))
#define gluScaleImage(format, wIn, hIn, typeIn, dataIn, wOut, hOut, typeOut, dataOut) glatter_gluScaleImage((format), (wIn), (hIn), (typeIn), (dataIn), (wOut), (hOut), (typeOut), (dataOut))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluScaleImage, (GLenum format, GLsizei wIn, GLsizei hIn, GLenum typeIn, const void *dataIn, GLsizei wOut, GLsizei hOut, GLenum typeOut, GLvoid* dataOut))
#define gluSphere(quad, radius, slices, stacks) glatter_gluSphere((quad), (radius), (slices), (stacks))
GLATTER_UBLOCK(void, GLAPIENTRY, gluSphere, (GLUquadric* quad, GLdouble radius, GLint slices, GLint stacks))
#define gluTessBeginContour(tess) glatter_gluTessBeginContour((tess))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessBeginContour, (GLUtesselator* tess))
#define gluTessBeginPolygon(tess, data) glatter_gluTessBeginPolygon((tess), (data))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessBeginPolygon, (GLUtesselator* tess, GLvoid* data))
#define gluTessCallback(tess, which, CallBackFunc) glatter_gluTessCallback((tess), (which), (CallBackFunc))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessCallback, (GLUtesselator* tess, GLenum which, _GLUfuncptr CallBackFunc))
#define gluTessEndContour(tess) glatter_gluTessEndContour((tess))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessEndContour, (GLUtesselator* tess))
#define gluTessEndPolygon(tess) glatter_gluTessEndPolygon((tess))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessEndPolygon, (GLUtesselator* tess))
#define gluTessNormal(tess, valueX, valueY, valueZ) glatter_gluTessNormal((tess), (valueX), (valueY), (valueZ))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessNormal, (GLUtesselator* tess, GLdouble valueX, GLdouble valueY, GLdouble valueZ))
#define gluTessProperty(tess, which, data) glatter_gluTessProperty((tess), (which), (data))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessProperty, (GLUtesselator* tess, GLenum which, GLdouble data))
#define gluTessVertex(tess, location, data) glatter_gluTessVertex((tess), (location), (data))
GLATTER_UBLOCK(void, GLAPIENTRY, gluTessVertex, (GLUtesselator* tess, GLdouble *location, GLvoid* data))
#define gluUnProject(winX, winY, winZ, model, proj, view, objX, objY, objZ) glatter_gluUnProject((winX), (winY), (winZ), (model), (proj), (view), (objX), (objY), (objZ))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluUnProject, (GLdouble winX, GLdouble winY, GLdouble winZ, const GLdouble *model, const GLdouble *proj, const GLint *view, GLdouble* objX, GLdouble* objY, GLdouble* objZ))
#endif // !defined(__GLU_H__)
#define gluUnProject4(winX, winY, winZ, clipW, model, proj, view, nearVal, farVal, objX, objY, objZ, objW) glatter_gluUnProject4((winX), (winY), (winZ), (clipW), (model), (proj), (view), (nearVal), (farVal), (objX), (objY), (objZ), (objW))
GLATTER_UBLOCK(GLint, GLAPIENTRY, gluUnProject4, (GLdouble winX, GLdouble winY, GLdouble winZ, GLdouble clipW, const GLdouble *model, const GLdouble *proj, const GLint *view, GLdouble nearVal, GLdouble farVal, GLdouble* objX, GLdouble* objY, GLdouble* objZ, GLdouble* objW))
#endif // defined(__glu_h__)
#endif // GLATTER_GLU

