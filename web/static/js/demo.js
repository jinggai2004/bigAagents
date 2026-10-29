/* 演示标注的唯一实现（文档 §3.5）
 *
 * 规则：未接通 API 的数值区块带 `data-demo`（+ 可读名 `data-demo-name`）；
 * 页脚由本文件统一汇总成"演示：…"。某个区块接通后只要删掉该元素上的
 * `data-demo` 属性，页脚自动少一项 —— 不需要改 JS，也不会出现"页脚说演示、
 * 实际已接真数据"的对不上。
 */
(function (global) {
  'use strict';

  function collect(root) {
    var out = [], seen = {};
    var nodes = (root || document).querySelectorAll('[data-demo]');
    for (var i = 0; i < nodes.length; i++) {
      var name = nodes[i].getAttribute('data-demo-name') || nodes[i].getAttribute('data-demo');
      if (!name || seen[name]) continue;
      seen[name] = 1;
      out.push(name);
    }
    return out;
  }

  function report(el, root) {
    if (!el) return [];
    var list = collect(root);
    el.textContent = list.length ? ('演示 ' + list.join(' · ')) : '无演示数据';
    return list;
  }

  global.bigaDemo = { collect: collect, report: report };
})(window);
