/* 轻量 markdown → HTML（复盘正文、助手回答共用）。
 *
 * 为什么自己写而不引第三方：只需要报告里真实会出现的语法
 * （标题/引用/有序无序列表/表格/分隔线/粗体/行内代码/链接），
 * 而引入 marked/markdown-it 会多一个外网或打包资产，且默认允许裸 HTML。
 * 这里**先转义再套标签**，任何 <script> 都到不了 DOM。
 *
 * 用法：`el.innerHTML = bigaMd(src)`
 */
(() => {
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, (m, a) => '<code>' + a + '</code>')
      .replace(/\*\*([^*]+)\*\*/g, (m, a) => '<strong>' + a + '</strong>')
      .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
        (m, t, u) => '<a href="' + u + '" target="_blank" rel="noopener">' + t + '</a>');
  }

  function render(src) {
    const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let i = 0, list = null;
    const closeList = () => { if (list) { out.push('</' + list + '>'); list = null; } };
    while (i < lines.length) {
      const t = lines[i].trim();
      if (!t) { closeList(); i++; continue; }
      // 表格：首行 + 分隔行（| --- | :--: |）才认，避免把普通竖线当表
      if (t.startsWith('|') && i + 1 < lines.length && /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())) {
        closeList();
        const head = t.replace(/^\||\|$/g, '').split('|').map((c) => inline(esc(c.trim())));
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().startsWith('|')) {
          rows.push(lines[i].trim().replace(/^\||\|$/g, '').split('|').map((c) => inline(esc(c.trim()))));
          i++;
        }
        out.push('<table><thead><tr>' + head.map((h) => '<th>' + h + '</th>').join('') +
          '</tr></thead><tbody>' +
          rows.map((r) => '<tr>' + r.map((c) => '<td>' + c + '</td>').join('') + '</tr>').join('') +
          '</tbody></table>');
        continue;
      }
      let m;
      if ((m = t.match(/^(#{1,4})\s+(.*)$/))) {
        closeList();
        const lv = m[1].length;
        out.push('<h' + lv + '>' + inline(esc(m[2])) + '</h' + lv + '>');
      } else if (/^([-*_])\1{2,}$/.test(t.replace(/\s/g, ''))) {
        closeList(); out.push('<hr>');
      } else if (t.startsWith('>')) {
        closeList(); out.push('<blockquote>' + inline(esc(t.replace(/^>\s?/, ''))) + '</blockquote>');
      } else if ((m = t.match(/^[-*+]\s+(.*)$/))) {
        if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; }
        out.push('<li>' + inline(esc(m[1])) + '</li>');
      } else if ((m = t.match(/^\d+[.)]\s+(.*)$/))) {
        if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; }
        out.push('<li>' + inline(esc(m[1])) + '</li>');
      } else {
        closeList(); out.push('<p>' + inline(esc(t)) + '</p>');
      }
      i++;
    }
    closeList();
    return out.join('\n');
  }

  window.bigaMd = render;
  window.bigaEsc = esc;
})();
