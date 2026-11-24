// Modelo web: poner model.json y pesos .bin dentro de model/ (exportados desde notebook)
const ETIQUETAS=['paper','plastic','metal'];let modelo=null;
async function cargar(){if(!modelo)modelo=await tf.loadGraphModel('model/model.json');return modelo;}
async function inferir(img){const red=await cargar();const x=tf.tidy(()=>tf.browser.fromPixels(img).resizeBilinear([160,160]).toFloat().expandDims(0));
 // NOTA: el modelo de entrenamiento recibe píxeles transformados a [-1,1].
 // TODO P2.3: asegurar preprocessing equivalente al del notebook antes de llamar a red.
 let salida=await red.executeAsync(x);x.dispose();let tensor=Array.isArray(salida)?salida[0]:salida;
 const valores=Array.from(await tensor.data());(Array.isArray(salida)?salida:[salida]).forEach(t=>t.dispose());return valores;}
document.getElementById('clasificar').onclick=async()=>{
 const f=document.getElementById('archivo').files[0];const estado=document.getElementById('estado');
 if(!f || !['image/jpeg','image/png'].includes(f.type)){estado.textContent='Archivo inválido: seleccione JPG o PNG';return;}
 const url=URL.createObjectURL(f),img=document.getElementById('vista');img.src=url;
 try{await img.decode();const pred=await inferir(img);document.getElementById('resultados').textContent=JSON.stringify(Object.fromEntries(ETIQUETAS.map((n,i)=>[n,pred[i]])),null,2);estado.textContent='Predicción generada';}
 catch(e){estado.textContent='Error: '+e.message;}finally{URL.revokeObjectURL(url)}
};
