// Loads an example's wasm build, or explains why it can't run here.
export async function load(name) {
  if (!navigator.gpu) {
    const p = document.createElement('p');
    p.className = 'engine-error';
    p.textContent = 'This browser does not support WebGPU. Use a recent Chrome, Edge or Safari.';
    document.body.append(p);
    return;
  }
  const { default: init } = await import(`./${name}/pkg/${name}.js`);
  await init();
}
