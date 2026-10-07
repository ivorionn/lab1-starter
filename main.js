// CS405 · Lab 1 — your first triangle in WebGPU
// Work through the TODOs in order. After each one, check the matching
// checkpoint on the lab slides. The reference solution is in ../lab1-solution/.

// ---------------------------------------------------------------------------
// VERSION — leave exactly ONE line uncommented. Each extra builds on the last.
// ---------------------------------------------------------------------------
// const VERSION = 5;  // TODO 5: rotating square, aspect-correct, follows mouse
// const VERSION = 6;  // extra 1: regular N-gon, N from the slider
// const VERSION = 7;  // extra 2: + colour by distance from the centre
const VERSION = 8;     // extra 3: + rotation speed depends on mouse distance

// ---------------------------------------------------------------------------
// MATRIX — TODO 6: the same rotation, as a mat2x2f. Works with any VERSION.
// Leave exactly ONE line uncommented.
// ---------------------------------------------------------------------------
// const MATRIX = 0;  // TODO 5 as it was: cos/sin written out by hand
// const MATRIX = 1;  // checkpoint 6: R*p, turns exactly as before
const MATRIX = 2;     // R*S*p (left) next to S*R*p (right): AB != BA
// const MATRIX = 3;  // the classic bug: rows typed as columns = R^t, spins backwards

const canvas = document.querySelector('canvas');
if (!navigator.gpu) throw new Error('WebGPU not available');

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No GPU adapter found');
const device  = await adapter.requestDevice();

const ctx = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format, alphaMode: 'opaque' });
console.log('WebGPU ready:', format);

const SHADER = `
  // 32 bytes: angle, aspect, mouse.xy | n, version, matrix, (padding)
  struct U {angle: f32, aspect: f32, mouse: vec2f, n: f32, version: f32, matrix: f32, pad: f32};
  @group(0) @binding(0) var<uniform> u: U;

  const R = 0.4;          // polygon radius
  const TAU = 6.2831853;

  struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) colour: vec4f,
    @location(1) local: vec2f   // position before rotate/aspect/mouse
  };

  fn hue(h: f32) -> vec3f {
    return 0.5 + 0.5 * cos(TAU * (h + vec3f(0.0, 1.0 / 3.0, 2.0 / 3.0)));
  }

  @vertex fn vs(@builtin(vertex_index) i: u32,
                @builtin(instance_index) inst: u32)   // which copy: 0 or 1
       -> VSOut {
    var local: vec2f;
    var col: vec3f;

    if (u.version < 5.5) {
      // square = two triangles sharing the diagonal
      var p = array<vec2f, 6>(
        vec2f(-0.3, -0.3), vec2f( 0.3, -0.3), vec2f( 0.3,  0.3),
        vec2f(-0.3, -0.3), vec2f( 0.3,  0.3), vec2f(-0.3,  0.3));

      var c = array<vec3f, 6>(
        vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0),
        vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(1.0, 1.0, 0.0));

      local = p[i];
      col = c[i];
    } else {
      // N-gon as a triangle fan: triangle k = (centre, rim k, rim k+1)
      let tri = i / 3u;
      let k = i % 3u;
      if (k == 0u) {
        local = vec2f(0.0);
        col = vec3f(1.0);                         // white centre
      } else {
        let t = (f32(tri) + f32(k - 1u)) / u.n;   // 0..1 around the rim
        local = R * vec2f(cos(TAU * t), sin(TAU * t));
        col = hue(t);                             // rainbow rim
      }
    }

    let a = u.angle;
    let compare = u.matrix > 1.5 && u.matrix < 2.5;
    var q: vec2f;

    if (u.matrix < 0.5) {
      // TODO 5: the rotation written out by hand
      q = vec2f(local.x * cos(a) - local.y * sin(a),
                local.x * sin(a) + local.y * cos(a));
    } else {
      // TODO 6: the same rotation, as a matrix. It is built once per vertex
      // HERE; on Thursday it is built once per object in JavaScript and sent
      // in the uniform buffer.
      // mat2x2f takes COLUMNS: first column (cos, sin), second (-sin, cos).
      var R = mat2x2f(cos(a), sin(a), -sin(a), cos(a));
      if (u.matrix > 2.5) {
        // the numbers typed row by row: that is R^t = rotation by -a
        R = mat2x2f(cos(a), -sin(a), sin(a), cos(a));
      }
      let S = mat2x2f(1.4, 0.0, 0.0, 0.6);   // stretch x, squash y

      if (!compare) {
        q = R * local;           // checkpoint 6: turns exactly as before
      } else if (inst == 0u) {
        q = R * S * local;       // read right to left: S first, then R
      } else {
        q = S * R * local;       // R first, then S: the shape shears as it turns
      }
    }

    q.x /= u.aspect;   // undo the canvas stretch (after rotating)
    if (compare) {
      // two copies side by side instead of following the mouse
      if (inst == 0u) { q.x -= 0.5; } else { q.x += 0.5; }
    } else {
      q += u.mouse;    // move to the mouse (already in clip space)
    }

    var out: VSOut;
    out.pos = vec4f(q, 0.0, 1.0);
    out.colour = vec4f(col, 1.0);
    out.local = local;
    return out;
  }

  @fragment fn fs(in: VSOut) -> @location(0) vec4f {
    if (u.version > 6.5) {
      // true per-pixel distance (interpolating a per-vertex distance would be
      // wrong: the middle of each rim edge is closer than R)
      let d = length(in.local) / R;               // 0 at centre, 1 at corners
      return vec4f(mix(vec3f(1.0, 0.85, 0.3), vec3f(0.6, 0.1, 0.5), d), 1.0);
    }
    return in.colour;
  }` ;

const module = device.createShaderModule({ code: SHADER });

const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: { module, entryPoint: 'vs' },
  fragment: { module, entryPoint: 'fs', targets: [{ format }] }
});


const ubuf = device.createBuffer({
  size: 32,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const bind = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: ubuf } }]
});

// ---------------------------------------------------------------------------
// TODO 5 — square (two triangles), correct aspect ratio, follows the mouse.
// Extras — slider for N, distance colour, mouse-controlled speed.
// ---------------------------------------------------------------------------

const slider = document.querySelector('#n');
const nLabel = document.querySelector('#nval');
slider.parentElement.style.visibility = VERSION >= 6 ? 'visible' : 'hidden';
slider.addEventListener('input', () => { nLabel.textContent = slider.value; });
document.querySelector('#pair').style.display = MATRIX === 2 ? 'flex' : 'none';

let mouse = [0, 0];
canvas.addEventListener('pointermove', (e) => {
  const r = canvas.getBoundingClientRect();
  mouse = [
    ((e.clientX - r.left) / r.width) * 2 - 1,
    1 - ((e.clientY - r.top) / r.height) * 2,   // screen y goes down, clip y goes up
  ];
});

function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
}
window.addEventListener('resize', resize);
resize();

// The angle is accumulated (angle += speed * dt) instead of angle = speed * t,
// otherwise changing the speed would make the shape jump.
let angle = 0;
let last = performance.now();

function frame() {
  const now = performance.now();
  const dt = (now - last) * 0.001;
  last = now;

  let speed = 1;                                     // rad/s
  if (VERSION >= 8) {
    const dist = Math.hypot(mouse[0], mouse[1]);     // 0 at canvas centre
    speed = 0.3 + 4 * dist;
  }
  angle += speed * dt;

  const n = Number(slider.value);
  device.queue.writeBuffer(ubuf, 0, new Float32Array([
    angle, canvas.width / canvas.height, mouse[0], mouse[1],
    n, VERSION, MATRIX, 0,
  ]));

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{
    view: ctx.getCurrentTexture().createView(),
    clearValue: { r: 0.19, g: 0.2, b: 0.6, a: 1 },
    loadOp: 'clear', storeOp: 'store' }] });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  // N-gon = N triangles; MATRIX 2 draws 2 instances (copies) of it
  pass.draw(VERSION >= 6 ? 3 * n : 6, MATRIX === 2 ? 2 : 1);
  pass.end();

  device.queue.submit([enc.finish()]);

  requestAnimationFrame(frame);
}
frame();
