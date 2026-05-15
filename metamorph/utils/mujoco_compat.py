from dataclasses import dataclass

import numpy as np

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    mujoco = None
    HAS_MUJOCO = False

try:
    import mujoco_py

    HAS_MUJOCO_PY = True
except ImportError:
    mujoco_py = None
    HAS_MUJOCO_PY = False


if HAS_MUJOCO:
    BACKEND = "mujoco"
elif HAS_MUJOCO_PY:
    BACKEND = "mujoco_py"
else:
    BACKEND = None


def require_backend():
    if BACKEND is None:
        raise ImportError(
            "Neither `mujoco` nor `mujoco_py` is installed. "
            "Install the modern package with `pip install mujoco`."
        )


def get_backend():
    require_backend()
    return BACKEND


@dataclass
class SimState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray | None = None
    udd_state: object | None = None


def _obj_type(type_):
    mapping = {
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
    }
    if type_ not in mapping:
        raise ValueError("type_ {} is not supported.".format(type_))
    return mapping[type_]


def _all_names(model, type_):
    obj_type = _obj_type(type_)
    count_attr = {
        "site": "nsite",
        "geom": "ngeom",
        "body": "nbody",
        "sensor": "nsensor",
        "joint": "njnt",
        "camera": "ncam",
    }[type_]
    num = getattr(model, count_attr)
    names = []
    for idx in range(num):
        name = mujoco.mj_id2name(model, obj_type, idx)
        names.append(name if name is not None else "")
    return tuple(names)


class ModelAdapter:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        if name == "body_names":
            return self.body_names
        if name == "geom_names":
            return self.geom_names
        if name == "site_names":
            return self.site_names
        if name == "joint_names":
            return self.joint_names
        if name == "sensor_names":
            return self.sensor_names
        return getattr(self._model, name)

    @property
    def body_names(self):
        return _all_names(self._model, "body")

    @property
    def geom_names(self):
        return _all_names(self._model, "geom")

    @property
    def site_names(self):
        return _all_names(self._model, "site")

    @property
    def joint_names(self):
        return _all_names(self._model, "joint")

    @property
    def sensor_names(self):
        return _all_names(self._model, "sensor")

    def site_name2id(self, name):
        return mujoco.mj_name2id(self._model, _obj_type("site"), name)

    def geom_name2id(self, name):
        return mujoco.mj_name2id(self._model, _obj_type("geom"), name)

    def body_name2id(self, name):
        return mujoco.mj_name2id(self._model, _obj_type("body"), name)

    def sensor_name2id(self, name):
        return mujoco.mj_name2id(self._model, _obj_type("sensor"), name)

    def site_id2name(self, id_):
        return mujoco.mj_id2name(self._model, _obj_type("site"), id_)

    def geom_id2name(self, id_):
        return mujoco.mj_id2name(self._model, _obj_type("geom"), id_)

    def body_id2name(self, id_):
        return mujoco.mj_id2name(self._model, _obj_type("body"), id_)

    def sensor_id2name(self, id_):
        return mujoco.mj_id2name(self._model, _obj_type("sensor"), id_)

    def camera_name2id(self, name):
        return mujoco.mj_name2id(self._model, _obj_type("camera"), name)

    def get_joint_qpos_addr(self, joint_name):
        joint_id = mujoco.mj_name2id(self._model, _obj_type("joint"), joint_name)
        start = self._model.jnt_qposadr[joint_id]
        if joint_id + 1 < self._model.njnt:
            end = self._model.jnt_qposadr[joint_id + 1]
        else:
            end = self._model.nq
        return start if end == start + 1 else (start, end)

    def get_joint_qvel_addr(self, joint_name):
        joint_id = mujoco.mj_name2id(self._model, _obj_type("joint"), joint_name)
        start = self._model.jnt_dofadr[joint_id]
        if joint_id + 1 < self._model.njnt:
            end = self._model.jnt_dofadr[joint_id + 1]
        else:
            end = self._model.nv
        return start if end == start + 1 else (start, end)


class DataAdapter:
    def __init__(self, model, data):
        self._model = model
        self._data = data

    def __getattr__(self, name):
        if name == "body_xpos":
            return self._data.xpos
        if name == "body_xquat":
            return self._data.xquat
        if name == "body_xvelp":
            return self._body_velocity()[0]
        if name == "body_xvelr":
            return self._body_velocity()[1]
        return getattr(self._data, name)

    def _body_velocity(self):
        linear = np.zeros((self._model.nbody, 3), dtype=np.float64)
        angular = np.zeros((self._model.nbody, 3), dtype=np.float64)
        velocity = np.zeros(6, dtype=np.float64)
        for body_id in range(self._model.nbody):
            mujoco.mj_objectVelocity(
                self._model,
                self._data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity,
                0,
            )
            angular[body_id] = velocity[:3]
            linear[body_id] = velocity[3:]
        return linear, angular

    def get_body_xpos(self, name):
        body_id = mujoco.mj_name2id(self._model, _obj_type("body"), name)
        return self._data.xpos[body_id]

    def get_body_xmat(self, name):
        body_id = mujoco.mj_name2id(self._model, _obj_type("body"), name)
        return self._data.xmat[body_id]

    def get_site_xpos(self, name):
        site_id = mujoco.mj_name2id(self._model, _obj_type("site"), name)
        return self._data.site_xpos[site_id]


class MjSimAdapter:
    def __init__(self, model):
        self._model = model
        self._data = mujoco.MjData(model)
        self.model = ModelAdapter(model)
        self.data = DataAdapter(model, self._data)

    def step(self):
        mujoco.mj_step(self._model, self._data)

    def forward(self):
        mujoco.mj_forward(self._model, self._data)

    def reset(self):
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

    def get_state(self):
        act = None
        if getattr(self._data, "act", None) is not None:
            act = self._data.act.copy()
        return SimState(
            time=float(self._data.time),
            qpos=self._data.qpos.copy(),
            qvel=self._data.qvel.copy(),
            act=act,
            udd_state=None,
        )

    def set_state(self, state):
        self._data.time = state.time
        self._data.qpos[:] = state.qpos
        self._data.qvel[:] = state.qvel
        if state.act is not None and getattr(self._data, "act", None) is not None:
            self._data.act[:] = state.act
        mujoco.mj_forward(self._model, self._data)


class OffscreenRenderer:
    def __init__(self, sim):
        self.sim = sim
        self.renderer = None
        self.width = None
        self.height = None
        self._last_pixels = None
        self.cam = type("CameraConfig", (), {})()

    def _ensure_renderer(self, width, height):
        if self.renderer is None or self.width != width or self.height != height:
            self.renderer = mujoco.Renderer(self.sim._model, height=height, width=width)
            self.width = width
            self.height = height

    def render(self, width, height, camera_id=None):
        self._ensure_renderer(width, height)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.sim._data, camera=camera_id)
        self._last_pixels = self.renderer.render().copy()

    def read_pixels(self, width, height, depth=False):
        self._ensure_renderer(width, height)
        if depth:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.sim._data)
            depth_pixels = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            return None, depth_pixels
        if self._last_pixels is None:
            self.render(width, height)
        return self._last_pixels

    def close(self):
        self.renderer = None


def load_model_from_xml(xml_str):
    require_backend()
    if BACKEND == "mujoco":
        return mujoco.MjModel.from_xml_string(xml_str)
    return mujoco_py.load_model_from_xml(xml_str)


def make_sim(model):
    require_backend()
    if BACKEND == "mujoco":
        return MjSimAdapter(model)
    return mujoco_py.MjSim(model)


def make_sim_state(sim, time, qpos, qvel, act=None, udd_state=None):
    if BACKEND == "mujoco":
        return SimState(time, qpos, qvel, act=act, udd_state=udd_state)
    return mujoco_py.MjSimState(time, qpos, qvel, act, udd_state)


def make_human_viewer(sim):
    if BACKEND == "mujoco":
        raise NotImplementedError(
            "Human viewer is not yet implemented for the modern `mujoco` backend "
            "in this compatibility layer. Use rgb_array rendering for now."
        )
    return mujoco_py.MjViewer(sim)


def make_offscreen_viewer(sim):
    if BACKEND == "mujoco":
        return OffscreenRenderer(sim)
    return mujoco_py.MjRenderContextOffscreen(sim, -1)
